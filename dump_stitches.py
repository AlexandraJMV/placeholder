"""
dump_stitches.py
────────────────
Muestra la SuperNetwork completa: stages intercalados con junctions.
Por cada stage se imprime una tabla IN→OUT por bloque; por cada
junction se listan los 9 stitches con canales, resolución y ops.

Uso:
    python dump_stitches.py
    python dump_stitches.py --plan network_plan.pkl --img_size 160
"""

import sys, os, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simple_poc.supernet import SuperNetwork, StitchLayer

# ── Monkey-patch: capturar src/dst ch y res antes de que se pierdan ──────────
_orig_stitch_init = StitchLayer.__init__

def _patched_stitch_init(self, src_cfg, dst_cfg, init_mode='ls', weight_matrix=None):
    self._src_ch  = src_cfg.get('ch',  None)
    self._src_res = src_cfg.get('res', None)
    self._dst_ch  = dst_cfg.get('ch',  None)
    self._dst_res = dst_cfg.get('res', None)
    _orig_stitch_init(self, src_cfg, dst_cfg, init_mode, weight_matrix)

StitchLayer.__init__ = _patched_stitch_init
# ─────────────────────────────────────────────────────────────────────────────


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _shape_str(ch, res) -> str:
    """[ch, res, res] con anchura fija para alineación."""
    if res is not None:
        return f"[{ch:3d}, {res:3d}, {res:3d}]"
    return f"[{ch:3d}]"


def _ops_str(sl: StitchLayer) -> str:
    ops = []
    for m in sl.op:
        t = type(m).__name__
        if t == 'AvgPool2d':   ops.append(f"AvgPool2d(s={m.stride})")
        elif t == 'Upsample':  ops.append(f"Upsample(×{m.scale_factor})")
        elif t == 'BatchNorm2d': ops.append(f"BN({m.num_features})")
        elif t == 'Conv2d':    ops.append(f"Conv1×1({m.in_channels}→{m.out_channels})")
        elif t == 'LeakyReLU': ops.append("LReLU")
        elif t == 'LayerNorm': ops.append(f"LN({m.normalized_shape})")
        elif t == 'Linear':    ops.append(f"Linear({m.in_features}→{m.out_features})")
        elif t == 'Flatten':   ops.append("Flatten")
        else:                  ops.append(t)
    return " → ".join(ops)


def _build_stage_shapes(model: SuperNetwork, img_size: int):
    """
    Reconstruye (ch, res) de entrada y salida de cada bloque en cada stage.

    Fuente de datos:
      - stage_in[0][k]  → siempre (3, img_size)
      - stage_out[j][k] → stitches[j][k][0]._src_ch / _src_res  (cualquier dst)
      - stage_in[j+1][k]→ stitches[j][0][k]._dst_ch / _dst_res  (cualquier src)
      - stage_out[last] → model.heads[k].in_features (post-GlobalPool; res=None)
    """
    n       = model.num_stages
    choices = model.choices_per_stage
    s_in    = [[None] * choices[i] for i in range(n)]
    s_out   = [[None] * choices[i] for i in range(n)]

    # Stage 0: imagen original
    for k in range(choices[0]):
        s_in[0][k] = (3, img_size)

    # Junctions intermedias
    for j, junc in enumerate(model.stitches):
        for src_idx in range(len(junc)):
            sl = junc[src_idx][0]
            s_out[j][src_idx] = (sl._src_ch, sl._src_res)
        for dst_idx in range(len(junc[0])):
            sl = junc[0][dst_idx]
            s_in[j + 1][dst_idx] = (sl._dst_ch, sl._dst_res)

    # Último stage: salida viene de las heads (después de GlobalPool)
    last = n - 1
    for k, head in enumerate(model.heads):
        s_out[last][k] = (head.in_features, None)

    return s_in, s_out


def dump_stitches(model: SuperNetwork, img_size: int = 160):
    stage_names = [[b.model_name for b in s] for s in model.plan.center2block]
    n_stages    = model.num_stages
    n_junc      = len(model.stitches)
    n_s, n_d    = len(model.stitches[0]), len(model.stitches[0][0])

    s_in, s_out = _build_stage_shapes(model, img_size)

    # ── Cabecera ───────────────────────────────────────────────────────────────
    print()
    print("═" * 84)
    print("  SUPERNET DUMP  —  stages + junctions")
    print(f"  Input   : {img_size}×{img_size} px  (3 ch RGB)")
    print(f"  Stages  : {n_stages}  |  Junctions : {n_junc}  |  "
          f"Stitches/junction : {n_s}×{n_d} = {n_s*n_d}")
    print("═" * 84)

    for i in range(n_stages):
        names = stage_names[i]

        # ── Tabla de bloques del stage ────────────────────────────────────────
        print()
        lbl = f"STAGE {i}  ({len(names)} blocks)"
        print(f"  ╔═ {lbl} {'═' * (74 - len(lbl))}╗")
        print(f"  ║  {'#':<4}  {'Backbone':<32}  {'IN shape':>13}    {'OUT shape'}")
        print(f"  ║  {'─'*4}  {'─'*32}  {'─'*13}    {'─'*13}")

        for k, name in enumerate(names):
            in_ch,  in_r  = s_in[i][k]  or (None, None)
            out_ch, out_r = s_out[i][k] or (None, None)

            in_s  = _shape_str(in_ch,  in_r)  if in_ch  is not None else "     ?"
            if i == n_stages - 1:
                # Último stage: resolución colapsada por GlobalPool
                out_s = f"{_shape_str(out_ch, None)}  ← GlobalPool" if out_ch else "?"
            else:
                out_s = _shape_str(out_ch, out_r) if out_ch is not None else "     ?"

            print(f"  ║  [{k}]   {name:<32}  {in_s}  →  {out_s}")

        print(f"  ╚{'═' * 80}╝")

        # ── Junction que sigue a este stage ───────────────────────────────────
        if i >= n_junc:
            continue

        junc      = model.stitches[i]
        src_names = stage_names[i]
        dst_names = stage_names[i + 1]
        total_p   = sum(_count_params(sl) for row in junc for sl in row)

        print()
        print(f"  ↕  JUNCTION {i}  (Stage {i} → Stage {i+1})"
              f"  ·  {n_s}×{n_d} = {n_s*n_d} stitches  ·  total params: {total_p:,}")
        print(f"  │")

        for si, row in enumerate(junc):
            for di, sl in enumerate(row):
                same = "  ✓ same" if src_names[si] == dst_names[di] else ""

                # Cadena de resolución
                if sl._src_res and sl._dst_res and sl._src_res != sl._dst_res:
                    r = f"{sl._src_res}×{sl._src_res} → {sl._dst_res}×{sl._dst_res}"
                elif sl._src_res:
                    r = f"{sl._src_res}×{sl._src_res}"
                else:
                    r = "?"

                print(f"  │  [{si}→{di}]  "
                      f"{src_names[si]:<26} → {dst_names[di]:<26}{same}")
                print(f"  │         shape : {sl._src_ch}ch → {sl.out_ch}ch  |  res : {r}  |  params : {_count_params(sl):,}")
                print(f"  │         ops   : {_ops_str(sl)}")
                print(f"  │")

        print(f"  {'─' * 80}")

    # ── Heads ──────────────────────────────────────────────────────────────────
    print()
    n_cls = model.heads[0].out_features
    print(f"  ╔═ HEADS  (GlobalPool → Linear → {n_cls} clases) {'═'*38}╗")
    print(f"  ║  {'#':<4}  {'Backbone':<32}  Linear                    Params")
    print(f"  ║  {'─'*4}  {'─'*32}  {'─'*25}    {'─'*8}")
    for k, head in enumerate(model.heads):
        name = stage_names[-1][k]
        print(f"  ║  [{k}]   {name:<32}  {head.in_features} → {head.out_features}"
              f"{'':>17}  {_count_params(head):,}")
    print(f"  ╚{'═' * 80}╝")

    # ── Resumen ────────────────────────────────────────────────────────────────
    t_stage  = sum(_count_params(s) for stage in model.stages for s in stage)
    t_stitch = sum(_count_params(sl) for junc in model.stitches for row in junc for sl in row)
    t_head   = sum(_count_params(h) for h in model.heads)

    print()
    print("═" * 84)
    print("  PARÁMETROS")
    print(f"  Stages   (backbones):            {t_stage:>14,}")
    print(f"  Stitches (inter-stage):          {t_stitch:>14,}")
    print(f"  Heads    (FC):                   {t_head:>14,}")
    print(f"  {'─'*50}")
    print(f"  TOTAL:                           {t_stage+t_stitch+t_head:>14,}")
    print("═" * 84)
    print()


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--plan',        type=str, default='network_plan.pkl')
    p.add_argument('--img_size',    type=int, default=160)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--init_mode',   type=str, default='random', choices=['random','ls'])
    return p.parse_args()

def main():
    args = parse_args()
    print(f"[dump_stitches]  plan={args.plan}  img={args.img_size}px  "
          f"classes={args.num_classes}  init={args.init_mode}")
    print()
    model = SuperNetwork(
        plan_path        = args.plan,
        num_classes      = args.num_classes,
        input_size       = args.img_size,
        stitch_init_mode = args.init_mode,
        matrices_path    = None,
    )
    dump_stitches(model, img_size=args.img_size)

if __name__ == '__main__':
    main()
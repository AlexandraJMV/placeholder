# =============================================================================
# supernet.py  — MERGED (original + refactored)
#
# From ORIGINAL (preserved):
#   - shapes_db (MODEL_INOUT_SHAPE.json) for native input channel detection
#   - StitchLayer structure and initialization logic
#
# From REFACTORED (added):
#   - AdaptiveGlobalPool: handles CNN (B,C,H,W) and Transformer (B,L,C)
#   - calibrate_bn(): BN recalibration per-path before evaluation
#   - sample_path(rng): isolated RNG to avoid corrupting global state
#   - _run_dummy_pass(): auto-corrects channel mismatches as last resort
#   - shapes_db is optional — falls back to auto-detection if JSON missing
#   - input_size used for spatial dims (not ImageNet scale from shapes_db)
# =============================================================================

import sys
import os
import pickle
import torch
import torch.nn as nn
import json
import math
import re

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from blocklize import MODEL_BLOCKS
from simlarity.feature_extraction import create_sub_network
from simlarity.utils import Block, Block_Assign


def get_block_type(model_name):
    transformers = ['vit', 'swin', 'deit', 'tnt']
    for t in transformers:
        if t in model_name.lower():
            return 'TRANS'
    return 'CNN'


class OutputUnwrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        if isinstance(out, tuple): out = out[0]
        if isinstance(out, dict): out = list(out.values())[0]
        if isinstance(out, tuple): out = out[0]
        return out


class AdaptiveGlobalPool(nn.Module):
    """Handles both CNN (B,C,H,W) and Transformer (B,L,C) outputs."""
    def forward(self, x):
        if x.dim() == 4:
            return x.mean(dim=[2, 3])
        elif x.dim() == 3:
            return x.mean(dim=1)
        elif x.dim() == 2:
            return x
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


class StitchLayer(nn.Module):
    def __init__(self, src_cfg, dst_cfg, init_mode='ls', weight_matrix=None):
        super().__init__()
        self.mode   = f"{src_cfg['type']}-to-{dst_cfg['type']}"
        self.in_ch  = src_cfg['ch']
        self.out_ch = dst_cfg['ch']
        in_res  = src_cfg.get('res', None)
        out_res = dst_cfg.get('res', None)

        layers = []

        # Spatial resolution adjustment
        if in_res is not None and out_res is not None:
            if in_res > out_res:
                stride = in_res // out_res
                if stride > 1:
                    layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
            elif in_res < out_res:
                scale = out_res // in_res
                layers.append(nn.Upsample(scale_factor=scale, mode='nearest'))

        if self.mode in ['CNN-to-CNN', 'CNN-to-TRANS']:
            layers.append(nn.BatchNorm2d(self.in_ch))
            self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.conv)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            if self.mode == 'CNN-to-TRANS':
                layers.append(nn.Flatten(2))

        elif self.mode == 'TRANS-to-CNN':
            if out_res is not None:
                class Unflatten(nn.Module):
                    def __init__(self, h, w):
                        super().__init__()
                        self.h, self.w = h, w
                    def forward(self, x):
                        B, L, C = x.shape
                        return x.transpose(1, 2).reshape(B, C, self.h, self.w)
                layers.append(Unflatten(out_res, out_res))
            layers.append(nn.BatchNorm2d(self.in_ch))
            self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.conv)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        elif self.mode == 'TRANS-to-TRANS':
            layers.append(nn.LayerNorm(self.in_ch))
            self.conv   = None
            self.linear = nn.Linear(self.in_ch, self.out_ch)
            layers.append(self.linear)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        else:
            raise NotImplementedError(f"Unknown stitch mode: {self.mode}")

        self.op = nn.Sequential(*layers)
        self._apply_init(init_mode, weight_matrix)

    def _apply_init(self, mode, weight_matrix):
        target = getattr(self, 'conv', None) or getattr(self, 'linear', None)
        if target is None:
            return
        if mode == 'random':
            nn.init.kaiming_normal_(target.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        elif mode == 'ls':
            if weight_matrix is not None:
                with torch.no_grad():
                    w = weight_matrix
                    if target.weight.dim() == 4 and w.dim() == 2:
                        w = w.view(self.out_ch, self.in_ch, 1, 1)
                    target.weight.copy_(w)
            else:
                nn.init.orthogonal_(target.weight)
                with torch.no_grad():
                    target.weight.mul_(math.sqrt(2))

    def forward(self, x):
        return self.op(x)


class SuperNetwork(nn.Module):
    def __init__(self, plan_path, num_classes=10, input_size=32,
                 stitch_init_mode='ls', matrices_path=None):
        super().__init__()
        print(f">>> Assembling SuperNetwork (init={stitch_init_mode}, input={input_size})...")

        self.stitch_init_mode = stitch_init_mode
        self.input_size       = input_size
        self.matrices         = {}

        self.freeze_backbone = False
        if stitch_init_mode == 'ls' and matrices_path and os.path.exists(matrices_path):
            print(f"   Loading transform matrices from {matrices_path}...")
            with open(matrices_path, 'rb') as f:
                self.matrices = pickle.load(f)

        with open(plan_path, 'rb') as f:
            self.plan = pickle.load(f)

        self.num_stages       = len(self.plan.center2block)
        self.choices_per_stage = [len(opts) for opts in self.plan.center2block]
    
        self.stages   = nn.ModuleList()
        self.stitches = nn.ModuleList()

        # Load shapes_db — gives native input channels per block (optional)
        shapes_db      = {}
        shapes_db_path = "tools/MODEL_INOUT_SHAPE.json"
        if os.path.exists(shapes_db_path):
            with open(shapes_db_path) as f:
                shapes_db = json.load(f)
            print(f"   shapes_db loaded ✅")
        else:
            print(f"   ⚠️  shapes_db not found — will auto-detect input channels")

        stage_infos = []
        dummy_input = torch.randn(1, 3, input_size, input_size)

        for i, stage_blocks in enumerate(self.plan.center2block):
            print(f"  -> Building Stage {i} ({len(stage_blocks)} choices)...")
            current_stage_modules = nn.ModuleList()
            current_stage_infos   = []

            for block_obj in stage_blocks:
                try:
                    raw_sub   = self._extract_submodule(block_obj)
                    sub_model = OutputUnwrapper(raw_sub)
                    b_type    = get_block_type(block_obj.model_name)

                    # ── Get native input channels ────────────────────────────
                    # shapes_db is the reliable source; fallback to inspection
                    if shapes_db and block_obj.model_name in shapes_db:
                        start_node = MODEL_BLOCKS[block_obj.model_name][block_obj.node_list[0]]
                        raw_shape  = shapes_db[block_obj.model_name]['in_size'][start_node]
                        if len(raw_shape) == 4:
                            real_in_ch = raw_shape[1]
                        else:
                            real_in_ch = raw_shape[0]
                    else:
                        real_in_ch, _ = self._get_native_in_channels(raw_sub)

                    # Blocks starting at node 0 receive the raw image
                    if block_obj.node_list[0] == 0:
                        real_in_ch = 3

                    # Always use actual input_size for spatial (not ImageNet scale)
                    real_in_res = input_size

                    # ── Stage 0: add stem stitch if block doesn't expect 3ch ─
                    if i == 0 and real_in_ch != 3:
                        src_cfg = {'type': 'CNN', 'ch': 3,           'res': input_size}
                        dst_cfg = {'type': b_type, 'ch': real_in_ch, 'res': input_size}
                        matrix  = self._get_matrix('input', block_obj.model_name)
                        stem    = StitchLayer(src_cfg, dst_cfg,
                                             init_mode=self.stitch_init_mode,
                                             weight_matrix=matrix)
                        sub_model   = nn.Sequential(stem, sub_model)
                        real_in_ch  = 3
                        real_in_res = input_size

                    current_stage_modules.append(sub_model)

                    # ── Dummy forward to discover output shape ───────────────
                    # _run_dummy_pass auto-corrects channel mismatches
                    out_t, real_in_ch = self._run_dummy_pass(
                        sub_model, real_in_ch, real_in_res, b_type)

                    info = {
                        'type':       b_type,
                        'in_ch':      real_in_ch,
                        'in_res':     real_in_res,
                        'model_name': block_obj.model_name,
                    }
                    if out_t.dim() == 4:
                        info['out_ch']  = out_t.shape[1]
                        info['out_res'] = out_t.shape[2]
                    elif out_t.dim() == 3:
                        info['out_ch']  = out_t.shape[2]
                        info['out_res'] = int(out_t.shape[1] ** 0.5)
                        info['seq_len'] = out_t.shape[1]
                    else:
                        raise ValueError(f"Unexpected output dim {out_t.dim()} "
                                         f"from {block_obj.model_name}")

                    current_stage_infos.append(info)
                    print(f"     Block {block_obj.model_name}: "
                          f"in_ch={real_in_ch}, out_shape={tuple(out_t.shape[1:])}")

                except Exception as e:
                    print(f"  ERROR in block {block_obj.model_name}: {e}")
                    raise

            self.stages.append(current_stage_modules)
            stage_infos.append(current_stage_infos)

        # Build stitching layers
        for i in range(self.num_stages - 1):
            print(f"  -> Stitching Stage {i} -> Stage {i+1}...")
            stage_stitches = nn.ModuleList()
            src_infos = stage_infos[i]
            dst_infos = stage_infos[i + 1]

            for src_info in src_infos:
                row_stitches = nn.ModuleList()
                for dst_info in dst_infos:
                    s_cfg = {'type': src_info['type'],
                             'ch':   src_info['out_ch'],
                             'res':  src_info.get('out_res')}
                    d_cfg = {'type': dst_info['type'],
                             'ch':   dst_info['in_ch'],
                             'res':  dst_info.get('in_res')}
                    matrix = self._get_matrix(src_info['model_name'],
                                              dst_info['model_name'])
                    stitch = StitchLayer(s_cfg, d_cfg,
                                         init_mode=self.stitch_init_mode,
                                         weight_matrix=matrix)
                    row_stitches.append(stitch)
                stage_stitches.append(row_stitches)
            self.stitches.append(stage_stitches)

        self.global_pool = AdaptiveGlobalPool()

        self.heads = nn.ModuleList()
        for info in stage_infos[-1]:
            self.heads.append(nn.Linear(info['out_ch'], num_classes))

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable    = sum(p.numel() for p in self.parameters()
                           if p.requires_grad) / 1e6
        print(f">>> SuperNetwork ready — {total_params:.1f}M total, "
              f"{trainable:.1f}M trainable")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_native_in_channels(self, module):
        """Fallback: inspect first Conv2d or Linear for native input channels."""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m.in_channels, 'CNN'
            if isinstance(m, nn.Linear):
                return m.in_features, 'TRANS'
        return 3, 'CNN'

    def _run_dummy_pass(self, sub_model, in_ch, in_res, b_type):
        """
        Run dummy forward to get output shape.
        Auto-corrects channel mismatches by reading the expected count
        from PyTorch's RuntimeError (handles ShuffleNet splits, etc.)
        """
        for _ in range(8):
            try:
                if b_type == 'CNN':
                    test_in = torch.randn(1, in_ch, in_res, in_res)
                else:
                    test_in = torch.randn(1, in_res * in_res, in_ch)
                with torch.no_grad():
                    out_t = sub_model(test_in)
                return out_t, in_ch
            except RuntimeError as e:
                match = re.search(r'to have (\d+) channels, but got (\d+)', str(e))
                if match:
                    in_ch = int(match.group(1))
                else:
                    raise
        raise RuntimeError("Could not resolve native input channels after 8 attempts")

    def _get_matrix(self, src_name, dst_name):
        if not self.matrices:
            return None
        return self.matrices.get(f"{src_name}->{dst_name}", None)

    def _extract_submodule(self, block_obj):
        model_name = block_obj.model_name
        torchvision_models = ['resnet', 'squeezenet', 'densenet', 'shufflenet']
        if any(x in model_name for x in torchvision_models):
            import torchvision.models as tvm
            try:
                base_model = getattr(tvm, model_name)(weights='DEFAULT')
            except TypeError:
                base_model = getattr(tvm, model_name)(pretrained=True)
        else:
            import timm
            base_model = timm.create_model(model_name, pretrained=True)

        all_nodes        = MODEL_BLOCKS[model_name]
        start_idx        = block_obj.node_list[0]
        end_idx          = block_obj.node_list[-1]
        input_node_name  = all_nodes[start_idx]
        output_node_name = all_nodes[end_idx]
        input_args       = [input_node_name] if start_idx > 0 else []
        return create_sub_network(base_model, input_args, [output_node_name])

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x, path=None):
        if path is None:
            path = [torch.randint(0, c, (1,)).item()
                    for c in self.choices_per_stage]

        out = self.stages[0][path[0]](x)

        for i in range(1, self.num_stages):
            prev_idx = path[i - 1]
            curr_idx = path[i]
            out = self.stitches[i - 1][prev_idx][curr_idx](out)
            out = self.stages[i][curr_idx](out)

        out = self.global_pool(out)
        out = self.heads[path[-1]](out)
        return out

    # ------------------------------------------------------------------ #
    # BN Calibration
    # ------------------------------------------------------------------ #

    ###
    @torch.no_grad()
    def calibrate_bn(self, loader, path, n_batches=50, device='cuda'):
        """
        Recalibrate BN running statistics for a specific sub‑network path.
        FIX: Use torch.inference_mode() for safety and disable autocast.
        """
        self.set_bn_tracking(True)
        def reset_bn(m):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.reset_running_stats()
        self.apply(reset_bn)

        was_training = self.training
        self.train()
        # FIX: Use inference_mode for faster forward, autocast disabled already
        with torch.inference_mode(False):   # keep grads off but allow BN stat updates
            for batch_idx, (inputs, _) in enumerate(loader):
                if batch_idx >= n_batches:
                    break
                self(inputs.to(device), path=path)

        if not was_training:
            self.eval()
        # Tracking remains enabled – ready for eval()

    def set_bn_tracking(self, track: bool):
        """Enable or disable running stats tracking for all BN layers."""
        def _set_tracking(module):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.track_running_stats = track
        self.apply(_set_tracking)

    def set_backbone_requires_grad(self, requires_grad: bool):
        """Set requires_grad for all parameters inside self.stages (the backbone blocks)."""
        for param in self.stages.parameters():
            param.requires_grad = requires_grad
        self.freeze_backbone = not requires_grad
 
 
    def sample_path(self, rng=None):
        """Sample a random path. Uses isolated RNG if provided."""
        if rng is not None:
            return [rng.randint(0, c - 1) for c in self.choices_per_stage]
        return [torch.randint(0, c, (1,)).item() for c in self.choices_per_stage]


if __name__ == '__main__':
    pkl_path = "network_plan.pkl"
    if os.path.exists(pkl_path):
        net = SuperNetwork(pkl_path, num_classes=10, input_size=32,
                           stitch_init_mode='ls')
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        print(f"Output shape: {y.shape}")  # Should be (2, 10)
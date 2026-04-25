"""
validate_network_plan.py
========================
Production-grade validation of a DeRy `Block_Assign` .pkl artifact.

Validates every structural, set-theoretic, and tensor-algebraic constraint
imposed by the DeRy partition algorithm (Yang et al., NeurIPS 2022).

Constraint references (paper Section 3.2 / 3.3):
  Eq.(2-5):  cover-set clustering with K equivalence sets
  Eq.(8-11): one block per set, one block per stage index per model
  Eq.(5):    |B_i^(k)| < (1+eps) * |M_i| / K

Usage
-----
  # Minimal (no shape JSON):
  python validate_network_plan.py --plan network_plan.pkl

  # Full (includes tensor boundary checks):
  python validate_network_plan.py --plan network_plan.pkl \
      --shape_path tools/MODEL_INOUT_SHAPE.json \
      --K 4 --eps 0.2
"""

import argparse
import json
import logging
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dery.validate")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL METADATA  (ground truth – must match block_meta.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_ZOO: List[str] = [
    "resnet18",
    "mobilenetv3_small_050",
    "efficientnet_b0",
]

# Maps each model to its ordered list of atomic nodes.
# Index i → position i in the line-graph G(V,E).
MODEL_BLOCKS: Dict[str, List[str]] = {
    "resnet18": [
        "layer1.0", "layer1.1",
        "layer2.0", "layer2.1",
        "layer3.0", "layer3.1",
        "layer4.0", "layer4.1",
    ],
    "mobilenetv3_small_050": [
        "blocks.0.0",
        "blocks.1.0", "blocks.1.1",
        "blocks.2.0", "blocks.2.1", "blocks.2.2",
        "blocks.3.0", "blocks.3.1",
        "blocks.4.0", "blocks.4.1", "blocks.4.2",
        "blocks.5.0",
    ],
    "efficientnet_b0": [
        "blocks.0.0",
        "blocks.1.0", "blocks.1.1",
        "blocks.2.0", "blocks.2.1",
        "blocks.3.0", "blocks.3.1", "blocks.3.2",
        "blocks.4.0", "blocks.4.1", "blocks.4.2",
        "blocks.5.0", "blocks.5.1", "blocks.5.2", "blocks.5.3",
        "blocks.6.0",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Accumulates all findings from a single validation pass."""
    check_name: str
    passed: bool = True
    violations: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.violations.append(msg)
        log.error("    ✗  %s", msg)

    def ok(self, msg: str) -> None:
        log.info("    ✓  %s", msg)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PICKLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_plan(pkl_path: str) -> Any:
    """
    Deserialise a Block_Assign pickle.

    The pickle was written by `partition.py` whose imports reside under the
    `simlarity` (typo preserved) namespace.  We inject minimal stubs so that
    pickle can reconstruct the object graph without requiring the full repo.
    """
    import types

    # ── Stub: blocklize package ──────────────────────────────────────────────
    blocklize_pkg = types.ModuleType("blocklize")
    blocklize_pkg.MODEL_ZOO    = MODEL_ZOO
    blocklize_pkg.MODEL_BLOCKS = MODEL_BLOCKS
    blocklize_pkg.MODEL_PRINT  = {m: m for m in MODEL_ZOO}
    blocklize_pkg.MODEL_STATS  = {m: {} for m in MODEL_ZOO}
    sys.modules.setdefault("blocklize", blocklize_pkg)

    blocklize_bm = types.ModuleType("blocklize.block_meta")
    blocklize_bm.MODEL_ZOO         = MODEL_ZOO
    blocklize_bm.MODEL_BLOCKS      = MODEL_BLOCKS
    blocklize_bm.MODEL_PRINT       = {m: m for m in MODEL_ZOO}
    blocklize_bm.MODEL_STATS       = {m: {} for m in MODEL_ZOO}
    blocklize_bm.MODEL_INOUT_SHAPE = None
    sys.modules.setdefault("blocklize.block_meta", blocklize_bm)

    # ── Stub: simlarity package (note typo is intentional – matches source) ──
    simlarity_pkg = types.ModuleType("simlarity")
    sys.modules.setdefault("simlarity", simlarity_pkg)

    # Minimal Block / Block_Assign / Block_Sim classes so pickle can re-bind
    # __reduce__ references without importing the real codebase.
    class Block:
        def __init__(self, model_name, block_index, node_list):
            self.model_name  = model_name
            self.block_index = block_index
            self.node_list   = node_list
            self.value       = 0
            self.size        = 0
            self.group_id    = None
        def __len__(self):   return len(self.node_list)
        def __eq__(self, o): return (isinstance(o, Block) and
                                     self.model_name  == o.model_name and
                                     self.block_index == o.block_index and
                                     self.node_list   == o.node_list)
        def __repr__(self):  return (f"Block({self.model_name}, "
                                     f"stage={self.block_index}, "
                                     f"nodes={self.node_list})")

    class Block_Sim:
        def __init__(self, sim_dict): self.sim_dict = sim_dict

    class Block_Assign:
        def __init__(self, assignment_index=None,
                     block_split_dict=None, centers=None):
            self.block2center = {}
            self.center2block = []
            self.centers      = centers or []

    simlarity_utils = types.ModuleType("simlarity.utils")
    simlarity_utils.Block        = Block
    simlarity_utils.Block_Sim    = Block_Sim
    simlarity_utils.Block_Assign = Block_Assign
    sys.modules.setdefault("simlarity.utils", simlarity_utils)

    with open(pkl_path, "rb") as fh:
        plan = pickle.load(fh)

    return plan


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INDIVIDUAL CHECK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 4.1  Structural type / attribute check ────────────────────────────────────

def check_object_structure(plan: Any, K: int) -> ValidationResult:
    """
    Verify that the top-level Block_Assign object exposes exactly the three
    attributes written by `Block_Assign.__init__` and that their cardinalities
    are consistent with K.

    Expected invariants
    -------------------
      len(plan.centers)      == K
      len(plan.center2block) == K
      plan.block2center      is a dict keyed by model name
    """
    res = ValidationResult("object_structure")
    log.info("CHECK 1 – Object structure & attribute cardinality")

    required_attrs = ("centers", "center2block", "block2center")
    for attr in required_attrs:
        if not hasattr(plan, attr):
            res.fail(f"Block_Assign missing attribute '{attr}'")

    if res.passed:
        if len(plan.centers) != K:
            res.fail(f"len(centers)={len(plan.centers)} ≠ K={K}")
        else:
            res.ok(f"len(centers) == K == {K}")

        if len(plan.center2block) != K:
            res.fail(f"len(center2block)={len(plan.center2block)} ≠ K={K}")
        else:
            res.ok(f"len(center2block) == K == {K}")

        if not isinstance(plan.block2center, dict):
            res.fail(f"block2center type={type(plan.block2center).__name__}, expected dict")
        else:
            res.ok("block2center is dict")

    return res


# ── 4.2  Center membership ────────────────────────────────────────────────────

def check_center_membership(plan: Any, K: int) -> ValidationResult:
    """
    Each anchor block a_j (Eq. 2) must be present in its own equivalence set G_j,
    i.e. plan.centers[j] ∈ plan.center2block[j].

    This is trivially required for the K-Means assignment update in `recenter`
    and is enforced explicitly in `Block_Assign.__init__`.
    """
    res = ValidationResult("center_membership")
    log.info("CHECK 2 – Anchor block membership in its own equivalence set")

    for gi in range(K):
        center = plan.centers[gi]
        group  = plan.center2block[gi]
        if not any(b == center for b in group):
            res.fail(
                f"Group {gi}: anchor block "
                f"({center.model_name}, stage={center.block_index}, "
                f"nodes={center.node_list}) absent from center2block[{gi}]"
            )
        else:
            res.ok(
                f"Group {gi}: anchor ({center.model_name} "
                f"stage={center.block_index}) present"
            )

    return res


# ── 4.3  Stage completeness per model in block2center ─────────────────────────

def check_stage_completeness(plan: Any, K: int) -> ValidationResult:
    """
    For every model M_i, block2center must contain exactly K entries keyed by
    stage indices {0, 1, …, K-1}, mapping each stage to a center block.

    This implements the Eq.(11) constraint: Σ_j Y(ik,j) = 1 for each k.
    If any stage is missing, the partition did not cover all K sub-graphs of M_i.
    """
    res = ValidationResult("stage_completeness")
    log.info("CHECK 3 – Per-model stage completeness in block2center")

    for model in MODEL_ZOO:
        if model not in plan.block2center:
            res.fail(f"{model}: entirely absent from block2center")
            continue

        stages_found = sorted(plan.block2center[model].keys())
        expected     = list(range(K))

        if stages_found != expected:
            missing = sorted(set(expected) - set(stages_found))
            extra   = sorted(set(stages_found) - set(expected))
            res.fail(
                f"{model}: stages_found={stages_found} ≠ expected={expected} "
                f"| missing={missing}, extra={extra}"
            )
        else:
            res.ok(f"{model}: all K={K} stages present → {stages_found}")

    return res


# ── 4.4  Node contiguity within every block ───────────────────────────────────

def check_node_contiguity(plan: Any) -> ValidationResult:
    """
    In the DeRy line-graph G(V,E), a block B_i^(k) is defined as a *contiguous*
    sub-path.  Its node_list must therefore satisfy:

        ∀ t ∈ {0,…,|B|−2}:  node_list[t+1] − node_list[t] == 1

    A gap of size > 1 implies an orphaned atomic node that belongs to no block,
    violating the disjoint-cover requirement of Eq.(4):
        B_i^(1) ∘ B_i^(2) ∘ ⋯ ∘ B_i^(K) = M_i
    """
    res = ValidationResult("node_contiguity")
    log.info("CHECK 4 – Intra-block node contiguity (line-graph sub-path integrity)")

    all_blocks = [b for group in plan.center2block for b in group]
    for b in all_blocks:
        nl = b.node_list
        if len(nl) == 0:
            res.fail(
                f"({b.model_name}, stage={b.block_index}): "
                f"empty node_list – violates min_node≥1 invariant"
            )
            continue
        gaps = [
            (nl[t], nl[t + 1])
            for t in range(len(nl) - 1)
            if nl[t + 1] - nl[t] != 1
        ]
        if gaps:
            res.fail(
                f"({b.model_name}, stage={b.block_index}): "
                f"non-contiguous node_list={nl} – gaps at transitions {gaps}"
            )
        else:
            res.ok(
                f"({b.model_name}, stage={b.block_index}): "
                f"nodes={nl} contiguous"
            )

    return res


# ── 4.5  MECE partition (Mutually Exclusive & Collectively Exhaustive) ─────────

def check_mece_partition(plan: Any) -> ValidationResult:
    """
    Eq.(4) of the paper requires the partition to be disjoint and exhaustive:

        B_i^(k1) ∩ B_i^(k2) = ∅  ∀ k1 ≠ k2        (mutual exclusivity)
        ∪_{k=1}^{K} B_i^(k) = V_i                   (collective exhaustivity)

    where V_i is the full atomic-node set of model M_i.

    Here we detect three sub-violations:
      (a) Duplicate node assignment: the same node index covered by ≥2 blocks
          in center2block.  This is a topological error introduced when the
          KL repartition step moves a node to a new block but fails to remove
          it from the source.
      (b) Missing node: an atomic node index ∈ [0, N−1] not covered by any
          block.  This would create a disconnected sub-graph.
      (c) Duplicate (model, stage) registration: the same (model_name,
          block_index) pair appearing in center2block with different node_lists,
          which indicates that multiple independent trials placed inconsistent
          versions of the same partition segment into the same group.
    """
    res = ValidationResult("mece_partition")
    log.info("CHECK 5 – MECE node partition per model")

    for model in MODEL_ZOO:
        N = len(MODEL_BLOCKS[model])
        expected_nodes = set(range(N))

        # Collect all blocks for this model across ALL equivalence sets
        all_blocks = [
            b for gi, group in enumerate(plan.center2block)
            for b in group if b.model_name == model
        ]

        # (c) Duplicate (model, stage) fingerprint within center2block
        seen_fingerprints: Dict[int, List[Any]] = defaultdict(list)  # stage -> [blocks]
        for b in all_blocks:
            seen_fingerprints[b.block_index].append(b)

        for stage, blocks in sorted(seen_fingerprints.items()):
            if len(blocks) > 1:
                node_lists = [b.node_list for b in blocks]
                groups     = [
                    gi for gi, group in enumerate(plan.center2block)
                    for b in group
                    if b.model_name == model and b.block_index == stage
                ]
                res.fail(
                    f"{model} stage={stage}: appears {len(blocks)}× in "
                    f"center2block with node_lists={node_lists} "
                    f"in groups={groups}. "
                    f"Root cause: Block_Assign.__init__ does not de-duplicate "
                    f"blocks across concurrent trial iterations."
                )

        # (a) Duplicate node coverage
        node_occurrence: Dict[int, List[Any]] = defaultdict(list)
        for b in all_blocks:
            for n in b.node_list:
                node_occurrence[n].append(b)

        for node_idx, owners in sorted(node_occurrence.items()):
            if len(owners) > 1:
                owner_strs = [f"stage={o.block_index},nodes={o.node_list}" for o in owners]
                res.fail(
                    f"{model} node[{node_idx}]={MODEL_BLOCKS[model][node_idx]!r}: "
                    f"claimed by {len(owners)} blocks → [{'; '.join(owner_strs)}]. "
                    f"Violates B_i^(k1)∩B_i^(k2)=∅ (Eq. 4)."
                )

        # (b) Missing nodes
        covered = set(n for b in all_blocks for n in b.node_list)
        missing = expected_nodes - covered
        if missing:
            missing_names = [MODEL_BLOCKS[model][i] for i in sorted(missing)]
            res.fail(
                f"{model}: nodes {sorted(missing)} ({missing_names}) not covered "
                f"by any block in center2block. "
                f"Violates ∪B_i^(k)=M_i (Eq. 4)."
            )
        else:
            if not any(v for v in res.violations if model in v and "stage=" in v and "appears" in v):
                res.ok(f"{model}: all {N} nodes covered, no duplicates")

    return res


# ── 4.6  Block size constraint ────────────────────────────────────────────────

def check_block_size_constraint(plan: Any, K: int, eps: float) -> ValidationResult:
    """
    Eq.(5):  |B_i^(k)| < (1 + ε) · |M_i| / K

    max_node_per_block = ⌈|M_i|/K⌉ · (1 + ε)

    This uses the same formula as `init_partition` in partition.py.
    A violation means the KL repartition loop permitted a block to grow beyond
    its allowed capacity, either because the `max_node` guard in `repartition`
    was computed incorrectly (the previously fixed bug) or because the selected
    best trial already violated the constraint.

    We check ALL blocks present in center2block (including duplicates), so that
    duplicate-registration violations are surfaced here as well.
    """
    res = ValidationResult("block_size_constraint")
    log.info("CHECK 6 – Block size bound |B_i^(k)| ≤ ⌈N/K⌉·(1+ε)  [ε=%.2f]", eps)

    for model in MODEL_ZOO:
        N = len(MODEL_BLOCKS[model])
        max_allowed = math.ceil(N / K) * (1 + eps)

        blocks = [b for g in plan.center2block for b in g if b.model_name == model]
        for b in blocks:
            size = len(b.node_list)
            if size > max_allowed:
                res.fail(
                    f"{model} stage={b.block_index} nodes={b.node_list}: "
                    f"|B|={size} > max_allowed={max_allowed:.2f} "
                    f"(N={N}, K={K}, ε={eps}). "
                    f"Likely cause: max_node guard applied the wrong variable "
                    f"(pre-fix `len2<=max_node` bug in repartition())."
                )
            else:
                res.ok(
                    f"{model} stage={b.block_index}: "
                    f"|B|={size} ≤ {max_allowed:.2f} OK"
                )

    return res


# ── 4.7  block2center ↔ center2block bidirectional consistency ────────────────

def check_b2c_c2b_consistency(plan: Any, K: int) -> ValidationResult:
    """
    block2center[model][k] returns the center block c for model M_i stage k.
    That center c must be the anchor of exactly one equivalence set G_j, i.e.
    plan.centers[j] == c and plan.block2center[model][k].group_id == j.

    Cross-referencing these two data structures detects cases where the
    K-Means re-centring step updated `centers` but did not re-flush the
    `block2center` pointers (or vice versa), producing a structurally
    inconsistent assignment.

    We also verify that every block B_i^(k) recorded in block2center actually
    appears in the center2block list of its claimed group.
    """
    res = ValidationResult("b2c_c2b_consistency")
    log.info("CHECK 7 – block2center ↔ center2block bidirectional consistency")

    # Build reverse map: Block identity → group index (from center2block)
    # Uses (model_name, block_index, frozenset(node_list)) as key to tolerate
    # the duplicate-registration case without masking subsequent checks.
    c2b_lookup: Dict[Tuple, List[int]] = defaultdict(list)
    for gi, group in enumerate(plan.center2block):
        for b in group:
            key = (b.model_name, b.block_index, tuple(b.node_list))
            c2b_lookup[key].append(gi)

    for model in MODEL_ZOO:
        stage_dict = plan.block2center.get(model, {})
        for stage in range(K):
            if stage not in stage_dict:
                # Missing stage already caught by check_stage_completeness
                continue
            center = stage_dict[stage]
            claimed_group = center.group_id

            # Verify center is indeed the anchor of claimed_group
            if claimed_group is not None and claimed_group < K:
                anchor = plan.centers[claimed_group]
                if anchor != center:
                    res.fail(
                        f"{model} stage={stage}: block2center center has "
                        f"group_id={claimed_group} but plan.centers[{claimed_group}] "
                        f"= ({anchor.model_name}, stage={anchor.block_index}, "
                        f"nodes={anchor.node_list}) – pointer mismatch."
                    )

            # Verify the center block exists in center2block of claimed_group
            c_key = (center.model_name, center.block_index, tuple(center.node_list))
            groups_in_c2b = c2b_lookup.get(c_key, [])
            if not groups_in_c2b:
                res.fail(
                    f"{model} stage={stage}: center "
                    f"({center.model_name}, stage={center.block_index}, "
                    f"nodes={center.node_list}) absent from center2block entirely."
                )
            elif claimed_group not in groups_in_c2b:
                res.fail(
                    f"{model} stage={stage}: block2center claims center is in "
                    f"group {claimed_group}, but center2block places it in "
                    f"group(s) {groups_in_c2b}."
                )
            else:
                res.ok(
                    f"{model} stage={stage}: block2center↔center2block "
                    f"consistent for group {claimed_group}"
                )

    return res


# ── 4.8  Equivalence-set population (K-stage representativeness) ──────────────

def check_equivalence_set_coverage(plan: Any, K: int) -> ValidationResult:
    """
    Per the DeRy design, each equivalence set G_j should contain at least one
    block from every model so that the reassembly step (Eq. 10) has a valid
    candidate from each source for stage j.  An empty or singleton group
    indicates that the K-Means clustering degenerated — typically due to:
      1. All trials converging to the same local minimum (non_improved > 20
         triggers early exit before diversification).
      2. The initial random seed producing a cluster that pulls no blocks.
    """
    res = ValidationResult("equivalence_set_coverage")
    log.info("CHECK 8 – Equivalence-set multi-model representation")

    for gi in range(K):
        group = plan.center2block[gi]
        models_in_group = {b.model_name for b in group}
        missing_models  = set(MODEL_ZOO) - models_in_group
        size = len(group)

        if size == 0:
            res.fail(f"Group {gi}: EMPTY – no blocks assigned.")
        elif missing_models:
            res.fail(
                f"Group {gi} ({size} blocks): lacks representation from "
                f"{sorted(missing_models)}. "
                f"Present: {sorted(models_in_group)}. "
                f"This group cannot serve as a complete stage in reassembly."
            )
        else:
            res.ok(
                f"Group {gi} ({size} blocks): all {len(MODEL_ZOO)} models "
                f"represented"
            )

    return res


# ── 4.9  Tensor boundary compatibility (requires MODEL_INOUT_SHAPE.json) ───────

def check_tensor_boundaries(
    plan: Any,
    K: int,
    shape_data: Optional[Dict],
) -> ValidationResult:
    """
    For any two consecutive equivalence sets G_k → G_{k+1} in a reassembled
    network, the output tensor shape of the selected block from G_k must equal
    the input tensor shape of the block from G_{k+1}, or a 1×1 stitching
    layer must be inserted.

    Formally, for anchor blocks a_k and a_{k+1}:

        out_shape(a_k) == in_shape(a_{k+1})  →  no stitching required
        out_shape(a_k) != in_shape(a_{k+1})  →  1×1 Conv stitching REQUIRED

    We check the ANCHOR BLOCKS (centers) because they define the canonical
    interface for each equivalence set used in reassembly (Section 3.3).

    Shape encoding: MODEL_INOUT_SHAPE[model][in_size|out_size][node_name]
    = [C, H, W] for CNN or [T, D] for Transformer.
    """
    res = ValidationResult("tensor_boundaries")
    log.info("CHECK 9 – Tensor boundary compatibility across equivalence sets")

    if shape_data is None:
        log.warning(
            "    ⚠  MODEL_INOUT_SHAPE.json not provided – "
            "tensor boundary checks SKIPPED. "
            "Run count_inout_size.py and pass --shape_path to enable."
        )
        res.passed = True  # Not a failure; data is missing, not wrong.
        res.violations.append(
            "SKIPPED: shape_data is None. "
            "Re-run with --shape_path tools/MODEL_INOUT_SHAPE.json"
        )
        return res

    for gi in range(K - 1):
        center_k   = plan.centers[gi]
        center_k1  = plan.centers[gi + 1]

        # Resolve named output node of center_k
        try:
            end_node_name = MODEL_BLOCKS[center_k.model_name][center_k.node_list[-1]]
            out_shape     = shape_data[center_k.model_name]["out_size"][end_node_name]
        except (KeyError, TypeError) as exc:
            res.fail(
                f"G_{gi}→G_{gi+1}: cannot resolve out_shape of anchor "
                f"({center_k.model_name}, stage={center_k.block_index}): {exc}"
            )
            continue

        # Resolve named input node of center_k1
        try:
            start_node_name = MODEL_BLOCKS[center_k1.model_name][center_k1.node_list[0]]
            in_shape        = shape_data[center_k1.model_name]["in_size"][start_node_name]
        except (KeyError, TypeError) as exc:
            res.fail(
                f"G_{gi}→G_{gi+1}: cannot resolve in_shape of anchor "
                f"({center_k1.model_name}, stage={center_k1.block_index}): {exc}"
            )
            continue

        if list(out_shape) == list(in_shape):
            res.ok(
                f"G_{gi}→G_{gi+1}: "
                f"{center_k.model_name}[end={end_node_name}] "
                f"out={out_shape} == "
                f"{center_k1.model_name}[start={start_node_name}] "
                f"in={in_shape} – NO stitching required"
            )
        else:
            res.fail(
                f"G_{gi}→G_{gi+1}: STITCHING REQUIRED – "
                f"{center_k.model_name}[end={end_node_name}] "
                f"out={out_shape} ≠ "
                f"{center_k1.model_name}[start={start_node_name}] "
                f"in={in_shape}. "
                f"A 1×1 Conv adapter must be inserted at this boundary."
            )

    return res


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY REPORTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[ValidationResult]) -> bool:
    """
    Print a structured pass/fail table and return True iff all checks passed.
    """
    SEP = "─" * 72
    print(f"\n{SEP}")
    print(f"{'VALIDATION SUMMARY':^72}")
    print(SEP)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        mark   = "✓" if r.passed else "✗"
        print(f"  [{status}]  {mark}  {r.check_name}")
        for v in r.violations:
            prefix = "         SKIP" if v.startswith("SKIPPED") else "         ERR "
            print(f"{prefix}: {v}")
        if not r.passed:
            all_passed = False

    print(SEP)
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results
                  if r.passed and r.violations and
                  r.violations[0].startswith("SKIPPED"))
    failed  = total - passed

    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  "
          f"Skipped-checks: {skipped}")
    print(SEP + "\n")

    if all_passed:
        print("✓  Network plan is STRUCTURALLY VALID.\n")
    else:
        print("✗  Network plan has VIOLATIONS. "
              "Reassembly will produce incorrect or non-runnable networks.\n")

    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ARGUMENT PARSING & ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeRy network-plan (Block_Assign .pkl) structural validator"
    )
    p.add_argument(
        "--plan", required=True,
        help="Path to the Block_Assign .pkl file (output of partition.py)"
    )
    p.add_argument(
        "--shape_path", default=None,
        help="Path to MODEL_INOUT_SHAPE.json (output of count_inout_size.py). "
             "Required for tensor boundary checks (Check 9)."
    )
    p.add_argument(
        "--K", type=int, default=4,
        help="Number of partition stages (default: 4)"
    )
    p.add_argument(
        "--eps", type=float, default=0.2,
        help="Block-size tolerance ε in |B| ≤ ⌈N/K⌉·(1+ε)  (default: 0.2)"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show all check-level OK messages (default: errors only)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.verbose:
        log.setLevel(logging.WARNING)  # suppress per-check OK lines

    pkl_path = Path(args.plan)
    if not pkl_path.exists():
        log.error("Plan file not found: %s", pkl_path)
        sys.exit(2)

    log.warning("Loading plan: %s", pkl_path)
    plan = load_plan(str(pkl_path))
    log.warning("Plan loaded. Type=%s", type(plan).__name__)

    # Optional shape JSON
    shape_data: Optional[Dict] = None
    if args.shape_path:
        shape_path = Path(args.shape_path)
        if not shape_path.exists():
            log.error("shape_path not found: %s", shape_path)
            sys.exit(2)
        with open(shape_path) as fh:
            shape_data = json.load(fh)
        log.warning("Shape data loaded from %s", shape_path)

    log.warning("=" * 72)
    log.warning("Beginning validation: K=%d, ε=%.2f", args.K, args.eps)
    log.warning("=" * 72)

    results: List[ValidationResult] = [
        check_object_structure(plan,   args.K),
        check_center_membership(plan,  args.K),
        check_stage_completeness(plan, args.K),
        check_node_contiguity(plan),
        check_mece_partition(plan),
        check_block_size_constraint(plan, args.K, args.eps),
        check_b2c_c2b_consistency(plan, args.K),
        check_equivalence_set_coverage(plan, args.K),
        check_tensor_boundaries(plan, args.K, shape_data),
    ]

    valid = print_summary(results)
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()

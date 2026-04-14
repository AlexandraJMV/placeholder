# =============================================================================
# supernet.py  —  REFACTORED
# Changes:
#   - Fixed dummy-pass resolution tracking (FLAW 4)
#   - Added adaptive global pooling for Transformer outputs (FLAW 5)
#   - Added calibrate_bn() utility method (FLAW 1)
#   - Cleaner stage_infos tracking using actual forward-pass output shapes
# =============================================================================

import sys
import os
import pickle
import torch
import torch.nn as nn
import json
import math

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
    """
    FIX (FLAW 5): Handles both CNN (B,C,H,W) and Transformer (B,L,C) outputs.
    """
    def forward(self, x):
        if x.dim() == 4:
            # CNN output: (B, C, H, W) -> (B, C)
            return x.mean(dim=[2, 3])
        elif x.dim() == 3:
            # Transformer output: (B, L, C) -> (B, C)
            return x.mean(dim=1)
        elif x.dim() == 2:
            return x
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")


class StitchLayer(nn.Module):
    def __init__(self, src_cfg, dst_cfg, init_mode='ls', weight_matrix=None):
        super().__init__()
        self.mode = f"{src_cfg['type']}-to-{dst_cfg['type']}"
        self.in_ch = src_cfg['ch']
        self.out_ch = dst_cfg['ch']
        in_res = src_cfg.get('res', None)
        out_res = dst_cfg.get('res', None)

        layers = []

        # Spatial resolution adjustment (CNN paths only)
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
            self.proj = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.proj)
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
            self.proj = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.proj)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        elif self.mode == 'TRANS-to-TRANS':
            layers.append(nn.LayerNorm(self.in_ch))
            self.proj = nn.Linear(self.in_ch, self.out_ch)
            layers.append(self.proj)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        else:
            raise NotImplementedError(f"Unknown stitch mode: {self.mode}")

        self.op = nn.Sequential(*layers)
        self._apply_init(init_mode, weight_matrix)

    def _apply_init(self, mode, weight_matrix):
        proj = getattr(self, 'proj', None)
        if proj is None:
            return
        if mode == 'random':
            nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif mode == 'ls':
            if weight_matrix is not None:
                with torch.no_grad():
                    w = weight_matrix
                    if proj.weight.dim() == 4 and w.dim() == 2:
                        w = w.view(self.out_ch, self.in_ch, 1, 1)
                    proj.weight.copy_(w)
            else:
                nn.init.orthogonal_(proj.weight)
                with torch.no_grad():
                    proj.weight.mul_(math.sqrt(2))

    def forward(self, x):
        return self.op(x)


class SuperNetwork(nn.Module):
    def __init__(self, plan_path, num_classes=10, input_size=32,
                 stitch_init_mode='ls', matrices_path=None):
        super().__init__()
        print(f">>> Assembling SuperNetwork (init={stitch_init_mode}, input={input_size})...")

        self.stitch_init_mode = stitch_init_mode
        self.input_size = input_size
        self.matrices = {}

        if stitch_init_mode == 'ls' and matrices_path and os.path.exists(matrices_path):
            print(f"   Loading transform matrices from {matrices_path}...")
            with open(matrices_path, 'rb') as f:
                self.matrices = pickle.load(f)

        with open(plan_path, 'rb') as f:
            self.plan = pickle.load(f)

        self.num_stages = len(self.plan.center2block)
        self.choices_per_stage = [len(opts) for opts in self.plan.center2block]

        self.stages = nn.ModuleList()
        self.stitches = nn.ModuleList()

        # FIX (FLAW 4): Track actual output shapes via real dummy passes,
        # not from shapes_db which stores ImageNet scales.
        stage_infos = []

        # Stage 0: use real input
        dummy_per_block = {}
        dummy_input = torch.randn(1, 3, input_size, input_size)

        for i, stage_blocks in enumerate(self.plan.center2block):
            print(f"  -> Building Stage {i} ({len(stage_blocks)} choices)...")
            current_stage_modules = nn.ModuleList()
            current_stage_infos = []

            for block_idx, block_obj in enumerate(stage_blocks):
                try:
                    raw_sub = self._extract_submodule(block_obj)
                    sub_model = OutputUnwrapper(raw_sub)
                    b_type = get_block_type(block_obj.model_name)

                    # FIX (FLAW 4): Determine actual test input from previous stage
                    # output shapes, not from shapes_db.
                    if i == 0:
                        test_in = dummy_input
                        real_in_ch = 3
                        real_in_res = input_size
                    else:
                        # Use the actual output from the corresponding previous block
                        # (same index if available, else first block in prev stage)
                        prev_info = stage_infos[i-1][0]  
                        real_in_ch = prev_info['out_ch']
                        real_in_res = prev_info['out_res']
                        if prev_info['type'] == 'CNN':
                            test_in = torch.randn(1, real_in_ch, real_in_res, real_in_res)
                        else:
                            # Transformer: (B, L, C) where L = seq_len
                            seq_len = prev_info.get('seq_len', real_in_res * real_in_res)
                            test_in = torch.randn(1, seq_len, real_in_ch)

                    # Add stem stitch for stage 0 blocks that don't expect raw input
                    if i == 0 and (real_in_ch != 3 or real_in_res != input_size):
                        src_cfg = {'type': 'CNN', 'ch': 3, 'res': input_size}
                        dst_cfg = {'type': b_type, 'ch': real_in_ch, 'res': real_in_res}
                        matrix = self._get_matrix('input', block_obj.model_name)
                        stem = StitchLayer(src_cfg, dst_cfg,
                                           init_mode=self.stitch_init_mode,
                                           weight_matrix=matrix)
                        sub_model = nn.Sequential(stem, sub_model)
                        test_in = dummy_input

                    current_stage_modules.append(sub_model)

                    # Actual dummy forward to get real output shape
                    with torch.no_grad():
                        out_t = sub_model(test_in)

                    # Build info from actual output
                    info = {
                        'type': b_type,
                        'in_ch': real_in_ch,
                        'in_res': real_in_res,
                        'model_name': block_obj.model_name,
                    }
                    if out_t.dim() == 4:
                        info['out_ch'] = out_t.shape[1]
                        info['out_res'] = out_t.shape[2]
                    elif out_t.dim() == 3:
                        # Transformer: (B, L, C)
                        info['out_ch'] = out_t.shape[2]
                        info['out_res'] = int(out_t.shape[1] ** 0.5)
                        info['seq_len'] = out_t.shape[1]
                    else:
                        raise ValueError(f"Unexpected output dim {out_t.dim()} "
                                         f"from {block_obj.model_name}")

                    current_stage_infos.append(info)
                    print(f"     Block {block_obj.model_name}: "
                          f"out_shape={tuple(out_t.shape[1:])}")

                except Exception as e:
                    print(f"  ERROR in block {block_obj.model_name}: {e}")
                    raise

            self.stages.append(current_stage_modules)
            stage_infos.append(current_stage_infos)

        # Build stitching layers using actual shapes
        for i in range(self.num_stages - 1):
            print(f"  -> Stitching Stage {i} -> Stage {i+1}...")
            stage_stitches = nn.ModuleList()
            src_infos = stage_infos[i]
            dst_infos = stage_infos[i + 1]

            for src_info in src_infos:
                row_stitches = nn.ModuleList()
                for dst_info in dst_infos:
                    s_cfg = {
                        'type': src_info['type'],
                        'ch': src_info['out_ch'],
                        'res': src_info.get('out_res'),
                    }
                    d_cfg = {
                        'type': dst_info['type'],
                        'ch': dst_info['in_ch'],
                        'res': dst_info.get('in_res'),
                    }
                    matrix = self._get_matrix(src_info['model_name'],
                                              dst_info['model_name'])
                    stitch = StitchLayer(s_cfg, d_cfg,
                                         init_mode=self.stitch_init_mode,
                                         weight_matrix=matrix)
                    row_stitches.append(stitch)
                stage_stitches.append(row_stitches)
            self.stitches.append(stage_stitches)

        # FIX (FLAW 5): Use adaptive pool that handles both CNN and Transformer outputs
        self.global_pool = AdaptiveGlobalPool()

        self.heads = nn.ModuleList()
        for info in stage_infos[-1]:
            self.heads.append(nn.Linear(info['out_ch'], num_classes))

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad) / 1e6
        print(f">>> SuperNetwork ready — {total_params:.1f}M total, "
              f"{trainable:.1f}M trainable")

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

        all_nodes = MODEL_BLOCKS[model_name]
        start_idx = block_obj.node_list[0]
        end_idx = block_obj.node_list[-1]
        input_node_name = all_nodes[start_idx]
        output_node_name = all_nodes[end_idx]
        input_args = [input_node_name] if start_idx > 0 else []
        return create_sub_network(base_model, input_args, [output_node_name])

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

        # FIX (FLAW 5): handles (B,C,H,W) and (B,L,C)
        out = self.global_pool(out)
        out = self.heads[path[-1]](out)
        return out

    @torch.no_grad()
    def calibrate_bn(self, loader, path, n_batches=50, device='cuda'):
        """
        FIX (FLAW 1): Recalibrate BatchNorm running statistics for a specific
        path after supernet training. Must be called before evaluating any
        candidate architecture during the search phase.

        Args:
            loader:    DataLoader for calibration (train split recommended).
            path:      List[int] — the architecture path to calibrate.
            n_batches: Number of batches to use for calibration (~50 is sufficient).
            device:    Target device.
        """
        # Reset running stats for all BN layers touched by this path
        def reset_bn(module):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.reset_running_stats()
                module.momentum = None  # Use cumulative moving average during calib

        # Only reset BN on the blocks in this specific path
        self.stages[0][path[0]].apply(reset_bn)
        for i in range(1, self.num_stages):
            self.stitches[i - 1][path[i - 1]][path[i]].apply(reset_bn)
            self.stages[i][path[i]].apply(reset_bn)

        # Run in train mode so BN accumulates stats, but no grad
        was_training = self.training
        self.train()
        for batch_idx, (inputs, _) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            inputs = inputs.to(device)
            self(inputs, path=path)

        if not was_training:
            self.eval()

        # Restore default momentum
        def restore_momentum(module):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.momentum = 0.1

        self.apply(restore_momentum)

    def sample_path(self, rng=None):
        """Sample a random path. Uses a local RNG if provided to avoid
        corrupting the global random state (FIX FLAW 7)."""
        if rng is not None:
            return [rng.randint(0, c - 1) for c in self.choices_per_stage]
        return [torch.randint(0, c, (1,)).item() for c in self.choices_per_stage]


if __name__ == '__main__':
    pkl_path = "network_plan.pkl"
    if os.path.exists(pkl_path):
        net = SuperNetwork(pkl_path, num_classes=10, input_size=32,
                           stitch_init_mode='ls')
        x = torch.razndn(2, 3, 32, 32)
        y = net(x)
        print(f"Output shape: {y.shape}")  # Should be (2, 10)
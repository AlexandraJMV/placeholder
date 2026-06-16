import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# Minimal SuperNetwork construction (without external dependencies)
# ----------------------------------------------------------------------
class DummyBlock(nn.Module):
    """A tiny CNN block with BN to simulate a sub‑network."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MinimalSuperNet(nn.Module):
    """Two‑stage supernet with dummy blocks for rapid testing."""
    def __init__(self):
        super().__init__()
        self.choices_per_stage = [2, 2]  # 2 choices per stage

        # Stage 0: two different blocks
        self.stage0 = nn.ModuleList([
            DummyBlock(3, 8),
            DummyBlock(3, 16)
        ])

        # Stitch: 1x1 convs to align channels between stages
        self.stitch = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(8, 16, 1),   # block0 -> block0
                nn.Conv2d(8, 32, 1)    # block0 -> block1
            ]),
            nn.ModuleList([
                nn.Conv2d(16, 16, 1),  # block1 -> block0
                nn.Conv2d(16, 32, 1)   # block1 -> block1
            ])
        ])

        # Stage 1: two different blocks
        self.stage1 = nn.ModuleList([
            DummyBlock(16, 32),
            DummyBlock(32, 64)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)

    def forward(self, x, path):
        # path = [idx_stage0, idx_stage1]
        x = self.stage0[path[0]](x)
        x = self.stitch[path[0]][path[1]](x)
        x = self.stage1[path[1]](x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    # BN control methods (identical to your fixed version)
    def set_bn_tracking(self, track):
        def _set(module):
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.track_running_stats = track
        self.apply(_set)

    @torch.no_grad()
    def calibrate_bn(self, loader, path, n_batches=10, device='cpu'):
        self.set_bn_tracking(True)
        def reset_bn(m):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                m.reset_running_stats()
        self.apply(reset_bn)

        was_training = self.training
        self.train()
        for i, (inputs, _) in enumerate(loader):
            if i >= n_batches:
                break
            self(inputs.to(device), path=path)
        if not was_training:
            self.eval()

    def sample_path(self):
        return [torch.randint(0, c, (1,)).item() for c in self.choices_per_stage]

# ----------------------------------------------------------------------
# Test Harness
# ----------------------------------------------------------------------
def test_bn_handling():
    print("🧪 Testing BN handling fixes...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MinimalSuperNet().to(device)

    # Dummy data loaders (10 batches of 8 samples each)
    dummy_data = torch.randn(80, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (80,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 1. Simulate training with tracking disabled
    print("  → Simulating training (track_running_stats=False)...")
    model.train()
    model.set_bn_tracking(False)
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= 5:
            break
        x, y = x.to(device), y.to(device)
        path = model.sample_path()
        out = model(x, path)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
    print("    ✅ Training loop completed without BN tracking")

    # 2. Verify that running stats are still in initial state (untouched)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Running mean should be zeros, var ones (default)
            assert torch.allclose(module.running_mean, torch.zeros_like(module.running_mean)), \
                f"Running mean modified during training: {name}"
            assert torch.allclose(module.running_var, torch.ones_like(module.running_var)), \
                f"Running var modified during training: {name}"
    print("    ✅ BN running stats untouched during training")

    # 3. Test calibration for a specific path
    print("  → Testing BN calibration...")
    test_path = [0, 1]
    model.calibrate_bn(loader, test_path, n_batches=5, device=device)
    print("    ✅ Calibration completed without crashes")

    # 4. Verify that running stats have been updated
    any_updated = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if not (torch.allclose(module.running_mean, torch.zeros_like(module.running_mean)) and
                    torch.allclose(module.running_var, torch.ones_like(module.running_var))):
                any_updated = True
                break
    assert any_updated, "Running stats not updated after calibration!"
    print("    ✅ Running stats successfully updated")

    # 5. Test evaluation after calibration
    print("  → Testing evaluation with calibrated stats...")
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(4, 3, 32, 32).to(device)
        out = model(x_test, test_path)
    print("    ✅ Evaluation succeeded (no RuntimeError)")

    # 6. Ensure that track_running_stats remained True after calibration
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.track_running_stats is True, "track_running_stats not left as True!"
    print("    ✅ track_running_stats correctly preserved")

    # 7. Reset for next epoch (should disable tracking again)
    model.train()
    model.set_bn_tracking(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.track_running_stats is False, "Failed to disable tracking!"
    print("    ✅ Tracking toggled off for next epoch")

    print("\n🎉 All tests passed! BN fixes are working correctly.")

if __name__ == "__main__":
    test_bn_handling()
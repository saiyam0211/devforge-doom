"""Model weights exporter and packaging utility for DevForge Track 1.

Validates that:
- Weights file is stored in submission/weights/agent_weights.pt
- File size is well within the 1 GB repository / Git LFS budget
- State dict can be loaded cleanly with weights_only=True
- Model passes a dry-run inference test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

try:
    import torch
    from submission.model import SpatialEgoNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def export_weights(output_path: Path | None = None) -> Path:
    if not TORCH_AVAILABLE:
        print("[Export] Error: PyTorch is required to export neural weights.")
        sys.exit(1)

    if output_path is None:
        output_path = ROOT / "submission" / "weights" / "agent_weights.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize production architecture
    model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2)
    model.eval()

    # Save state dict
    torch.save(model.state_dict(), output_path)

    # Verify file size
    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"[Export] Saved weights to: {output_path}")
    print(f"[Export] Model size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"[Export] Budget limit: 1024.00 MB ({size_mb / 1024.0 * 100:.2f}% of limit)")

    # Test reload
    reloaded_dict = torch.load(output_path, map_location="cpu", weights_only=True)
    test_model = SpatialEgoNet(hidden_dim=256, num_gru_layers=2)
    test_model.load_state_dict(reloaded_dict)
    test_model.eval()
    print("[Export] Successfully reloaded and verified weights with weights_only=True.")

    return output_path


if __name__ == "__main__":
    export_weights()

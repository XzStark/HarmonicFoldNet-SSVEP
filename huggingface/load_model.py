from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from .harmonicfoldnet import HarmonicFoldNet, reparameterize_model
except ImportError:  # Direct execution from a downloaded model snapshot.
    from harmonicfoldnet import HarmonicFoldNet, reparameterize_model


def load_harmonic_fold_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    folded: bool = False,
) -> tuple[HarmonicFoldNet, dict[str, Any]]:
    """Load a released fold checkpoint without requiring the original EEG files."""

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    result = payload["result"]
    frequencies = state[
        "temporal_candidate_features.class_frequencies"
    ].detach().cpu().tolist()
    phases = state[
        "temporal_candidate_features.class_phases"
    ].detach().cpu().tolist()

    model = HarmonicFoldNet(
        channels=len(result["channels"]),
        sample_rate=int(payload["config"]["sample_rate"]),
        class_frequencies=frequencies,
        class_phases=phases,
        **result["model_config"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    if folded:
        model = reparameterize_model(model)
        model.eval()
    model.to(device)
    return model, payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a HarmonicFoldNet fold checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folded", action="store_true")
    args = parser.parse_args()
    loaded, metadata = load_harmonic_fold_checkpoint(
        args.checkpoint, device=args.device, folded=args.folded,
    )
    print(
        {
            "architecture": metadata["result"]["architecture"],
            "dataset": metadata["result"]["dataset"],
            "parameters": sum(parameter.numel() for parameter in loaded.parameters()),
            "folded": args.folded,
        }
    )

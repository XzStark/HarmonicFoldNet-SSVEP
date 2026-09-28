from __future__ import annotations

from pathlib import Path

import torch

from .model import FastSSVEPFusionNet, reparameterize_model


def load_checkpoint_model(path: str | Path, *, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    frequencies = state["class_frequencies"].tolist()
    config = checkpoint["config"]
    model = FastSSVEPFusionNet(
        channels=int(state["time_encoder.projection.0.weight"].shape[1]),
        classes=len(frequencies), sample_rate=int(config["sample_rate"]),
        class_frequencies=frequencies, **config["model"],
    )
    if checkpoint.get("deploy", False):
        model = reparameterize_model(model)
    model.load_state_dict(state)
    return model.eval().to(device), checkpoint


def export_deploy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    model, checkpoint = load_checkpoint_model(source)
    deploy = reparameterize_model(model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": deploy.state_dict(),
        "config": checkpoint["config"],
        "result": checkpoint.get("result"),
        "deploy": True,
        "source_checkpoint": Path(source).name,
    }, destination)
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(export_deploy_checkpoint(args.source, args.destination))

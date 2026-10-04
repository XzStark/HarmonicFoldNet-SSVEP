# LegacyFusionNet v0.1.0 release bundle

This directory is the frozen upload bundle for the noncommercial research
preview. The checkpoint files are intentionally excluded from normal Git
tracking by `*.pt`; upload them as release/model-host assets from this exact
directory instead of duplicating or regenerating them.

## Files

- `model.pt`: training-form PyTorch checkpoint
- `model_deploy.pt`: structurally reparameterized inference checkpoint
- `config.yaml`: exact training configuration
- `metrics.json`: training history and aggregate metrics
- `evaluation.json`: fixed, per-subject evaluation and batch-1 latency
- `SHA256SUMS.txt`: integrity hashes for the files above

The bundle does not contain raw EEG. See the repository root for licenses,
attribution, limitations and reproduction instructions.

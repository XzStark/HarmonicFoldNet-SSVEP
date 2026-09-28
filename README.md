# FastSSVEPFusionNet

FastSSVEPFusionNet is a compact research model for cross-subject, 40-class
steady-state visual evoked potential (SSVEP) decoding. It combines
frequency-candidate evidence with local-first time/frequency encoders, late
attention, and train-to-deploy structural reparameterization.

This repository is a **noncommercial source-available research release**, not
an OSI-approved open-source release. It is a reproducible v0.1 research
preview, not a peer-reviewed performance claim or a medical product.

Source repository: https://github.com/XzStark/FastSSVEPFusionNet

## Frozen v0.1 result

Dataset: Kim2025BetaRange / NEMAR `nm000127` v1.0.2.

- 40 participants, 6 sessions and 9,600 trials
- 40 stimulus classes
- 8 posterior channels: `PO7, PO3, POz, PO4, PO8, O1, Oz, O2`
- 5-second windows sampled at 250 Hz
- subject-disjoint split: train 1-32, validation 33-36, test 37-40
- 960 held-out test trials

| System | Validation Top-1 | Test Top-1 | Test Top-5 |
|---|---:|---:|---:|
| Fixed two-harmonic evidence | 52.40% | 79.06% | - |
| Learned candidate evidence | 66.25% | 88.23% | 95.73% |
| Full fusion model | 71.46% | **88.65%** | **95.73%** |

The full model has 890,570 training-graph parameters; the fused deploy graph
has 888,266 parameters. Algebraic equivalence of the reparameterized blocks is
covered by tests.

### Batch-1 model latency

These numbers measure model forward time after the complete EEG window is
available. They are not acquisition-to-feedback latency.

| Device | Training graph | Deploy graph | Reduction |
|---|---:|---:|---:|
| CUDA GPU | 8.01 ms | 5.20 ms | 35.0% |
| CPU, one thread | 8.37 ms | 6.01 ms | 28.2% |

Exact unrounded values and per-subject results are frozen in
`runs/kim2025_full_v1/evaluation.json`.

## What this result does not establish

- This is one predefined subject split, not leave-one-subject-out validation.
- The result uses 5-second windows; short-window accuracy remains to be tested.
- The four held-out test participants vary from 80.83% to 99.58% Top-1.
- Wearable/peri-auricular layouts, calibration burden and end-to-end device
  latency have not yet been validated.
- The repository does not support unrestricted thought or inner-speech
  decoding.

These limitations are intentionally part of the release so that the v0.1
artifact cannot be mistaken for the final paper evaluation.

## Reproduce

Create an environment, install the dependencies, and obtain the dataset from
its official host. Raw EEG is not redistributed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare per-participant shards with `src.kim2025_data` (see `--help`), then:

```powershell
.\.venv\Scripts\python.exe -m src.train_kim2025 --config configs/kim2025.yaml --run-dir runs/kim2025_full_v1
.\.venv\Scripts\python.exe -m src.checkpoint --source runs/kim2025_full_v1/model.pt --destination runs/kim2025_full_v1/model_deploy.pt
.\.venv\Scripts\python.exe -m src.evaluate_kim2025 --checkpoint runs/kim2025_full_v1/model.pt --output runs/kim2025_full_v1/evaluation.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See `python -m src.kim2025_data --help` for dataset preparation options and
`docs/TECHNICAL_REPORT_v0.1.md` for the full protocol.

## Release contents

- `src/`: preprocessing, training, evaluation and deploy export
- `configs/kim2025.yaml`: frozen v0.1 configuration
- `runs/kim2025_full_v1/`: checkpoint and machine-readable evaluation record
- `huggingface/`: staged model-card bundle
- `docs/TECHNICAL_REPORT_v0.1.md`: technical report source
- `docs/ENVIRONMENT_v0.1.md`: measured software, hardware and latency protocol
- `docs/COMPARISON_FIGURE_PROMPT_zh.md`: prompt for the progress graphic
- `docs/PUBLIC_RELEASE_POLICY.md`: publication-risk boundary

## Data and attribution

Kim2025BetaRange is hosted by NEMAR as `nm000127` and is distributed under CC
BY 4.0. Obtain it from the official host and cite the dataset paper. The model
uses structural-reparameterization ideas described in prior work, while this
repository contains an original one-dimensional EEG implementation and does
not use image-model weights. See `THIRD_PARTY_NOTICES.md`.

## License

- Source code: PolyForm Noncommercial License 1.0.0 (`LICENSE`)
- Model weights and authored documentation: CC BY-NC 4.0
  (`MODEL_LICENSE.md`)
- Dataset: its own CC BY 4.0 terms; raw data is not included

Commercial use requires separate written permission. Because the source-code
license restricts commercial use, describe the project as **noncommercial
source-available**, not OSI open source.

## Safety

This software is for research. It is not a medical device and must not be used
for diagnosis, treatment, emergency response or safety-critical control.

## Citation

Use `CITATION.cff` for this v0.1 artifact. A paper citation can replace it only
after a public manuscript or DOI exists.

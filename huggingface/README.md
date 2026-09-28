---
license: cc-by-4.0
library_name: pytorch
tags:
  - eeg
  - ssvep
  - brain-computer-interface
  - time-series
  - structural-reparameterization
---

# FastSSVEPFusionNet

FastSSVEPFusionNet is a compact research model for 40-class SSVEP decoding. It
combines configurable candidate-frequency evidence with local-first time and
frequency encoders, late attention, and train/deploy structural
reparameterization.

## Intended use

Research on synchronized SSVEP EEG recorded from posterior scalp channels. It
is not a medical device, does not decode unrestricted thoughts or inner speech,
and should not be used for diagnosis or safety-critical control.

## Training data

The initial checkpoint was trained on Kim2025BetaRange / NEMAR `nm000127`
v1.0.2 under CC BY 4.0. Raw EEG is not redistributed. Users should obtain the
dataset from its official host and retain the original citation and license.

## Evaluation

The fixed split used participants 1-32 for training, 33-36 for validation, and
37-40 for held-out testing. The first checkpoint reached 88.65% test Top-1 and
95.73% test Top-5 on 960 held-out trials. Individual test-participant Top-1
ranged from 80.83% to 99.58%, so the aggregate should not be interpreted as a
guarantee for a new wearer.

## Input

- Channels: PO7, PO3, POz, PO4, PO8, O1, Oz, O2.
- Sampling rate: 250 Hz.
- Window: 5 seconds / 1,250 samples.
- Preprocessing: 6-45 Hz band-pass and per-window channel standardization.
- Candidate frequencies are loaded from dataset metadata, not embedded as
  application-specific labels in the data pipeline.

## Loading a checkpoint

```python
from src.checkpoint import load_checkpoint_model

model, metadata = load_checkpoint_model("model_deploy.pt")
```

The release contains a training-form checkpoint for further research and a
structurally reparameterized checkpoint for lower-latency inference.

## Attribution

The local-first and structural-reparameterization design is inspired by Apple
FastViT (Vasu et al., ICCV 2023). The implementation adapts the design to 1-D
multichannel EEG and adds an EEG-specific candidate-frequency evidence path.
See `THIRD_PARTY_NOTICES.md` in the source repository.

## Limitations

The current result is one subject-disjoint split. Cross-fold evaluation,
shorter-window experiments, conventional SSVEP baselines, additional deep
baselines, and real wearable recordings are required before making stronger
claims. The author's current self-recorded session was not hardware-synchronized
to the stimulus and is therefore excluded from efficacy results.

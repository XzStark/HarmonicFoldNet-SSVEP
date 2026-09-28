---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - eeg
  - ssvep
  - brain-computer-interface
  - time-series
  - cross-subject
  - structural-reparameterization
---

# FastSSVEPFusionNet v0.1

FastSSVEPFusionNet is a compact research model for 40-class SSVEP decoding.
It combines frequency-candidate evidence, local-first time/frequency encoders,
late attention and train-to-deploy structural reparameterization.

This is a noncommercial research preview. It is not peer reviewed, is not a
medical product, and is not evidence of unrestricted thought or inner-speech
decoding.

Source code: https://github.com/XzStark/FastSSVEPFusionNet

## Model details

- Framework: PyTorch
- Training graph: 890,570 parameters
- Deploy graph: 888,266 parameters
- Input: float EEG tensor shaped `[batch, 8, 1250]`
- Sampling rate: 250 Hz
- Window: 5 seconds
- Channels: `PO7, PO3, POz, PO4, PO8, O1, Oz, O2`
- Output: 40 class logits

`model.pt` is the training-form checkpoint. `model_deploy.pt` contains the
structurally fused inference form.

## Training data

The checkpoint was trained on Kim2025BetaRange / NEMAR `nm000127` v1.0.2,
distributed by its authors under CC BY 4.0. Raw EEG is not included. Obtain the
dataset from its official host and retain its citation and license.

- 40 participants, 6 sessions, 9,600 trials
- train participants 1-32: 7,680 trials
- validation participants 33-36: 960 trials
- test participants 37-40: 960 trials
- preprocessing: 6-45 Hz filtering and per-window standardization

## Frozen evaluation

| System | Validation Top-1 | Test Top-1 | Test Top-5 |
|---|---:|---:|---:|
| Fixed two-harmonic evidence | 52.40% | 79.06% | - |
| Learned candidate evidence | 66.25% | 88.23% | 95.73% |
| Full fusion model | 71.46% | **88.65%** | **95.73%** |

Batch-1 model-forward latency after a full input window is available:

| Device | Training graph | Deploy graph | Reduction |
|---|---:|---:|---:|
| CUDA GPU | 8.01 ms | 5.20 ms | 35.0% |
| CPU, one thread | 8.37 ms | 6.01 ms | 28.2% |

Exact values are provided in `evaluation.json`.

## Intended use

- Reproduction of the frozen v0.1 experiment
- Noncommercial research on synchronized posterior-channel SSVEP EEG
- Study of compact time/frequency fusion and structural reparameterization

## Limitations and out-of-scope uses

- Only one predefined subject-disjoint split is reported.
- The model has not yet been validated with leave-one-subject-out testing.
- The reported window is 5 seconds; short-window performance is unknown.
- Test-participant Top-1 ranges from 80.83% to 99.58%.
- Peri-auricular/wearable electrodes and end-to-end hardware latency are not
  validated.
- Do not use for diagnosis, treatment, emergency response, safety-critical
  control, identity inference, covert monitoring or claims of thought reading.

## License

The model weights and this model card are licensed under CC BY-NC 4.0. Source
code is distributed separately under PolyForm Noncommercial 1.0.0. Commercial
use requires separate written permission. Dataset terms remain independent.

## Citation

Until a manuscript or DOI exists, cite the software artifact using the
repository `CITATION.cff`. Do not invent a paper title, venue or DOI.

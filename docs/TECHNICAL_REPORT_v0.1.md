# FastSSVEPFusionNet: A Physics-Guided and Structurally Reparameterized Baseline for Efficient Cross-Subject SSVEP Decoding

**Xiangzhe Kong**

Research preview v0.1 - 29 September 2026

## Abstract

This technical report presents an initial subject-disjoint evaluation of a
compact model for 40-class steady-state visual evoked potential (SSVEP)
decoding. The model combines three elements: candidate-frequency evidence
derived from stimulus frequencies and harmonics, parallel time- and
frequency-domain encoders with attention applied after local processing, and a
one-dimensional train-to-deploy structural reparameterization path. On a fixed
split of 40 participants and 9,600 trials, the full model reached 88.65% Top-1
and 95.73% Top-5 accuracy on 960 held-out trials. A fixed two-harmonic power
baseline reached 79.06% Top-1, while the learned candidate-frequency evidence
path alone reached 88.23%. Reparameterization preserved numerical outputs and
reduced batch-one model latency from 8.01 ms to 5.20 ms on an NVIDIA RTX 5060
Laptop GPU and from 8.37 ms to 6.01 ms on one CPU thread of an AMD Ryzen 9
8945HX. These results are a reproducible research checkpoint, not a claim of
state of the art. The current experiment uses a five-second EEG window and one
fixed participant split; short-window, leave-one-subject-out, statistical, and
real-device evaluations remain necessary.

## 1. Scope and research question

SSVEP decoding must balance recognition quality, calibration requirements, and
deployment cost. This work asks whether explicit candidate-frequency evidence
can be combined with learned time/frequency representations and an equivalent
deployment graph to improve a subject-disjoint baseline without creating a
large model. The intended scope is brief, optional BCI interaction and research
on wearable input. It is not intended as an always-on flicker interface, a
medical device, or a system for decoding unrestricted thoughts.

## 2. Dataset and protocol

The experiment uses Kim2025BetaRange / NEMAR `nm000127` v1.0.2, licensed under
CC BY 4.0. The dataset contains 40 healthy participants, six sessions and 240
trials per participant, and 40 joint frequency-phase-modulated targets in the
14.0-21.8 Hz range. The source EEG was acquired at 1,024 Hz.

This experiment uses the eight posterior channels PO7, PO3, POz, PO4, PO8, O1,
Oz, and O2. Each complete five-second stimulation interval is resampled to 250
Hz, band-limited to 6-45 Hz, and standardized per channel and window.

The participant-disjoint split is fixed before training:

| Split | Participants | Trials |
| --- | --- | ---: |
| Train | 1-32 | 7,680 |
| Validation | 33-36 | 960 |
| Held-out test | 37-40 | 960 |

No participant occurs in more than one split. This protocol tests an initial
cross-participant setting, but it is not a substitute for full
leave-one-subject-out evaluation.

## 3. Method

### 3.1 Candidate-frequency evidence

For each candidate class, the model resolves the fundamental stimulus
frequency and its first two harmonics to spectrum bins. It aggregates power
over posterior channels and harmonics, normalizes evidence across candidate
classes, and applies a small learned residual function over channel-harmonic
features. The resulting evidence is scaled by a learned non-negative factor
and added to the classifier logits. Frequencies are supplied through dataset
metadata rather than embedded as application labels in the data pipeline.

### 3.2 Time and frequency encoders

The time branch processes standardized EEG windows. The frequency branch
processes log power after removal of a smooth spectral envelope and interpolation
to a fixed number of bins. Each branch applies channel projection, local
depthwise mixing, downsampling, and late multi-head attention. The two branch
representations are concatenated and projected before classification.

### 3.3 Train-to-deploy reparameterization

Local token mixing is trained with three depthwise paths: a larger-kernel
convolution, a pointwise convolution, and an identity batch-normalization path.
At deployment, convolution and normalization parameters are algebraically
fused into one depthwise convolution. Numerical equivalence is checked before
the deployment checkpoint is accepted. The design principle is informed by
prior work on structural reparameterization, while the present implementation
uses original one-dimensional EEG operators and does not use image-model
weights.

## 4. Training configuration

| Setting | Value |
| --- | --- |
| Random seed | 20260929 |
| Batch size | 128 |
| Evaluation batch size | 256 |
| Maximum epochs | 60 |
| Early stopping patience | 10 |
| Learning rate | 0.0005 |
| Weight decay | 0.02 |
| Label smoothing | 0.05 |
| Encoder width | 64 |
| Local blocks per branch | 3 |
| Attention blocks per branch | 2 |
| Attention heads | 4 |
| Dropout | 0.15 |
| Spectral bins | 256 |

The best validation checkpoint is retained for evaluation.

## 5. Results

### 5.1 Recognition accuracy

| Method | Validation Top-1 | Test Top-1 | Test Top-5 |
| --- | ---: | ---: | ---: |
| Fixed two-harmonic power | 52.40% | 79.06% | Not measured |
| Learned candidate-frequency evidence | 66.25% | 88.23% | 95.73% |
| Full time/frequency fusion model | 71.46% | **88.65%** | **95.73%** |

On the held-out test set, the full model improves Top-1 accuracy by 9.59
percentage points over the fixed harmonic baseline. The learned evidence path
accounts for most of that gain, improving by 9.17 points. The full model adds
0.42 points over learned evidence on this test split. On validation participants,
the full model improves by 19.06 points over fixed harmonic power and 5.21
points over learned evidence.

The four held-out participant accuracies for the full model are 80.83%, 99.58%,
85.00%, and 89.17%. The spread is material and prevents the aggregate from
being treated as a guarantee for an unseen wearer.

### 5.2 Model size and compute latency

| Graph | Parameters | RTX 5060 Laptop GPU | Ryzen 9 8945HX, one CPU thread |
| --- | ---: | ---: | ---: |
| Training graph | 890,570 | 8.01 ms | 8.37 ms |
| Deployment graph | 888,266 | **5.20 ms** | **6.01 ms** |
| Relative latency reduction | - | **35.0%** | **28.2%** |

Latency is batch-one model compute after a complete EEG window is already
available. It excludes signal acquisition and therefore must not be described
as end-to-end interaction latency. Hardware, software stack, precision, and
thermal state can change these measurements. The training and deployment graph
passed numerical equivalence checks under the repository tolerance.

## 6. Interpretation

Three conclusions are supported by this first experiment:

1. Explicit candidate-frequency evidence is the strongest measured contributor
   under the fixed participant split.
2. The complete model provides an additional validation gain, but its small
   test-set gain over evidence alone requires repeated participant folds and
   statistical testing.
3. Structural reparameterization reduces model compute latency without changing
   accepted numerical outputs on the measured desktop CPU and GPU.

The experiment does not yet demonstrate superiority to published methods,
short-window operation, universal participant generalization, wearable-electrode
performance, or a reduction in battery consumption.

## 7. Planned evaluation

The next evaluation stage will use 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, and 5.0 second
windows; subject-wise folds followed by full leave-one-subject-out evaluation;
CCA, FBCCA, TRCA, TDCA, EEGNet, SSVEPformer, and time/frequency baselines; paired
participant-level statistics; and P50/P95 latency, memory, temperature, and
power measurements on a target edge device. Self-recorded EEG with uncertain
hardware synchronization and substantial drift is excluded from efficacy
claims.

## 8. Reproducibility and release

The research release contains source code, configuration, evaluation records,
and training/deployment checkpoints. Raw EEG data is not redistributed. Users
must obtain the dataset from its official source and comply with its license.
Source code is available for noncommercial purposes under the PolyForm
Noncommercial License 1.0.0. Model weights and documentation are available for
noncommercial use under CC BY-NC 4.0. This is a source-available noncommercial
research release rather than an OSI-approved open-source release.

Source repository: https://github.com/XzStark/FastSSVEPFusionNet

## 9. References

1. H. Kim, K. Won, M. Ahn, and S. C. Jun, “A 40-Class SSVEP Speller Dataset:
   Beta Range Stimulation for Low-Fatigue BCI Applications,” *Scientific Data*,
   2025. https://doi.org/10.1038/s41597-025-06032-2
2. J. Chen, Y. Zhang, Y. Pan, P. Xu, and C. Guan, “A Transformer-based deep
   neural network model for SSVEP classification,” arXiv:2210.04172, 2022.
   https://doi.org/10.48550/arXiv.2210.04172
3. Y. Dai et al., “A time-frequency feature fusion-based deep learning network
   for SSVEP frequency recognition,” *Frontiers in Neuroscience*, 2025.
   https://doi.org/10.3389/fnins.2025.1679451
4. P. K. A. Vasu, J. Gabriel, J. Zhu, O. Tuzel, and A. Ranjan, “FastViT: A Fast
   Hybrid Vision Transformer Using Structural Reparameterization,” ICCV, 2023.
   https://doi.org/10.1109/ICCV51070.2023.00532
5. Kim2025BetaRange / NEMAR nm000127 v1.0.2 dataset record.
   https://www.nemar.org/dataset/nm000127

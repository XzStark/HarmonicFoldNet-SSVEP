# HarmonicFoldNet paper evaluation protocol v1

Status: evaluation logic frozen before paper-scale runs; the multi-dataset
evidence set remains expandable without tuning the architecture on test data.

## Architecture mainline

The paper's proposed model is fixed as an original one-dimensional,
local-to-global hybrid Transformer for SSVEP decoding. Its defining design
constraints are:

1. foldable local depthwise-convolution processing precedes global
   token mixing;
2. hierarchical reduction compresses the temporal or candidate-aligned tokens
   before late-stage self-attention;
3. training-time overparameterized branches are algebraically folded into an
   equivalent deployment graph; and
4. the target is an accuracy-latency Pareto improvement, not parameter count
   or speed in isolation.

This is a transfer of design principles to one-dimensional EEG, not a reuse of
Apple weights, code, image operators, or VLM claims. Analytic CCA/FBCCA,
phase-aligned demodulation, and the 3,138-parameter residual scorer are retained
as priors, baselines, and ablations only. The frozen protocol-v4
`evidence_only` model is a strong baseline and is not the paper's main
architectural contribution.

## Claims under test

The paper-scale evaluation is allowed to support only the following claims if
the corresponding tests pass:

1. **Short-window decoding:** the method retains useful accuracy and
   information-transfer-rate behavior as the stimulation window decreases.
2. **Cross-subject generalization:** performance persists for unseen
   participants under subject-disjoint evaluation.
3. **Efficient deployment:** train-to-deploy reparameterization preserves
   accepted numerical outputs while improving measured batch-one latency and
   resource use.

The primary optimization target is fixed before final cross-validation as
0.4-1.2 seconds. Longer windows remain evaluation-only observations and do not
participate in checkpoint selection. This follows the development finding that
mixing 2-5 second objectives can obscure the sub-second question.

Failure of one claim does not justify hiding its result. The paper title,
abstract and contribution list must be narrowed to the claims actually
supported.

## Dataset and signal contract

- Development dataset: Kim2025BetaRange / NEMAR `nm000127` v1.0.2.
- Participants: 40; trials: 9,600; classes: 40.
- Channels: `PO7, PO3, POz, PO4, PO8, O1, Oz, O2`.
- Sampling rate after preprocessing: 250 Hz.
- Band-pass preprocessing: 6-45 Hz, applied to the continuous/session signal
  before epoch extraction. The paper's online-compatible main track uses a
  causal fourth-order SOS filter. Previously generated zero-phase results are
  retained only as an explicitly labelled offline comparison.
- Epoch origin: stimulus onset; no post-hoc onset shift.
- Registered windows: 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 3.0 and 5.0 seconds.
- Every cropped window is standardized independently per channel. A window is
  never standardized using samples that occur after its endpoint.

No trial, window from a trial, or participant may appear in more than one
train/validation/test role within a run.

This 40-participant dataset is not the entire paper evidence. Repeated folds or
random seeds measure estimator stability; they do not count as new biological
participants. The final validation matrix therefore includes independent
datasets acquired with different equipment and experimental conditions:

| Dataset | Participants | Role |
| --- | ---: | --- |
| Kim2025 beta-range | 40 | architecture development and beta-band stress test |
| Tsinghua Benchmark | 35 | established 40-target laboratory benchmark |
| BETA | 70 | 40-target data collected in a less controlled environment |
| Dong2023 | 59 | independently reserved external-dataset validation after architecture and analysis freeze |
| Wearable SSVEP, wet and dry | 102 | mixed-condition wearable-electrode test |

The five datasets contain 306 independent participants in total. Wet and dry
recordings from the same Wearable participant are paired conditions, not two
participants. Dataset-specific class sets, frequency ranges, latency offsets
and trial durations are preserved and disclosed. They are never silently
concatenated under an assumed common label space.

## Development and final evaluation

### Development screen

The existing participant-disjoint split is retained only for implementation
checks and fixed-protocol development:

- train: participants 1-32
- validation: participants 33-36
- test: participants 37-40

The architecture and optimizer configuration are frozen after this screen.
Development results are not the paper's principal generalization evidence.

### Full grouped cross-validation

Every dataset is evaluated with a fixed five-fold, subject-disjoint outer
cross-validation split. Each participant appears in the held-out test portion
exactly once per training seed. Validation participants are selected
deterministically from the corresponding outer-fold training pool using the
fixed split seed; all remaining participants form the model-fitting set. Test
data are evaluated only after checkpoint selection is complete. The exact
outer folds and validation sets are identical across model-training seeds.
Deep-model runs use seeds 20260929, 20260930 and 20260931.

The unit of statistical analysis is the participant, not the individual EEG
trial. Report per-participant values, mean, standard deviation, median,
interquartile range and participant-bootstrap 95% confidence intervals.

## Comparison tracks

Calibration-free methods and participant-calibrated methods must not be mixed
as though they used equal supervision.

### Calibration-free / unseen-participant track

- fixed harmonic-power reference
- standard CCA
- FBCCA
- EEGNet trained without target-participant labels
- SSVEPformer trained without target-participant labels
- one reproducible modern lightweight network selected and frozen before final
  testing
- the local-to-global HarmonicFoldNet model and frozen ablations

### Participant-calibrated track

TRCA, eTRCA, TRCA-R and TDCA are reported separately with the exact number of
target-participant calibration blocks or trials disclosed. Their reproduction
uses the conventional nine posterior channels (`Pz, PO5, PO3, POz, PO4, PO6,
O1, Oz, O2`), 250 Hz sampling, dataset-specific visual latency (140 ms for
Benchmark and 130 ms for BETA), the registered 6-90 Hz zero-phase/filter-bank
preprocessing, and leave-one-block-out evaluation where required by the source
protocol. Every learned comparator, including the proposed model, receives the
same target-participant calibration data in this track. These results are never
ranked in the same table as zero-target-participant calibration results.

## Required ablations

- frozen fixed harmonic and protocol-v4 analytic evidence baselines
- full local-to-global hybrid model
- hybrid model without the analytic evidence prior
- hybrid model without late attention
- hybrid model without the foldable local-convolution stages
- local-only model before late attention
- attention-only replacement at matched training protocol
- training graph versus algebraically folded deployment graph

The evidence branch computes a regularized canonical-correlation prior against
complete sine/cosine subspaces and adds a phase-aligned complex-demodulation
residual using the dataset's published JFPM phase metadata. It was introduced
after the initial power-only development model failed to match CCA/FBCCA on the
held-out Wearable split. The power-only results remain an exploratory negative
result; protocol version 4 is the candidate architecture used for subsequent
multi-dataset validation. The required `evidence_only` ablation determines
whether learned fusion adds value beyond the embedded classical prior.

The old protocol-v4 time/frequency fusion path did not establish a reliable
gain over `evidence_only` and is retained as a negative control. The redesigned
local-to-global model may be claimed as beneficial only if its frozen,
participant-level short-window grouped-cross-validation results improve the
analytic-evidence baseline and survive the required component ablations.

## Metrics

- Top-1 accuracy
- balanced accuracy
- Top-5 accuracy
- ITR in bits/minute, using `window_seconds + 0.5` seconds as selection time
- participant-level paired differences against every registered baseline
- paired permutation or Wilcoxon signed-rank tests with Holm correction
- participant-bootstrap 95% confidence intervals and an effect size

Accuracy and ITR must be reported together; maximizing ITR by shortening the
window while accuracy collapses is not a positive result.

## Deployment protocol

- batch size 1
- identical checkpoint before and after reparameterization
- equivalence assertion before benchmarking
- warm-up separated from timed iterations
- at least 1,000 timed forwards for desktop CPU/GPU reporting
- P50, P95, mean and standard deviation
- CPU thread count, precision, software versions and thermal state recorded
- peak memory recorded where the runtime exposes it
- sustained run used for temperature and power observations
- model compute reported separately from acquisition, preprocessing and full
  interaction latency

Desktop measurements support an implementation-efficiency claim. A mobile or
wearable deployment claim additionally requires measurement on the actual
target device.

## Multi-dataset validation

After development on Kim2025, freeze the architecture and repeat compatible
experiments on Tsinghua Benchmark, BETA and the paired wet/dry Wearable SSVEP
dataset. Any channel, frequency, phase, window, onset-latency or acquisition
mismatch must be documented. External datasets may expose a failure and lead
to a new model version, but data used to redesign that version cannot then be
reported as untouched external validation for the same version.

Report three distinct generalization questions rather than merging them into
one number:

1. within-dataset unseen-participant performance;
2. cross-dataset transfer without target-dataset fine-tuning;
3. target-dataset adaptation with the amount of calibration data disclosed.

The primary Wearable run pools dry- and wet-electrode trials within each
participant while keeping the outer split subject-disjoint. It therefore tests
unseen-participant performance under mixed acquisition conditions; it does not
by itself establish dry-to-wet or wet-to-dry transfer. Those two directional
transfers must be run as separate, explicitly filtered experiments before an
electrode-transfer claim is made. Even a positive result is only a public-data
proxy for the intended wearable setting, not a substitute for measurements
from the project's actual electrode placement and hardware.

## Release boundary

Intermediate checkpoints, incomplete folds and exploratory comparisons remain
private. Public figures are generated only from a complete result manifest.
No paper, preprint, release tag or public model update is made until the target
venue's prior-dissemination and anonymity rules have been checked.

# External baseline extension protocol v1

Frozen: 2026-10-09 (Asia/Shanghai)

## Status and scope

This is a post-result supplementary protocol. It was frozen after the primary
HarmonicFoldNet-versus-SSVEPformer results on Dong2023 had been inspected. It
therefore does not replace the preregistered primary analyses,
does not enter their multiplicity families, and must not be described as a
preregistered or confirmatory endpoint.

The purpose is narrower: add protocol-matched classical references on both
external datasets and one reproducible modern neural baseline under the same
data, channel, window, split, metric, and deployment-accounting rules.

## Frozen comparisons

### Classical unsupervised references

- Dataset: Dong2023 (59 participants).
- Methods: CCA and FBCCA.
- Participants: all available participants, so the estimates describe each
  external cohort rather than only the neural test fold.
- Channels, sampling rate, causal 6--45 Hz input shards, class frequencies,
  trial filters, and windows are inherited without alteration from
  `configs/paper_multidataset.yaml`.
- Reference bank: four harmonics.
- FBCCA filter bank: lower cutoffs 6, 12, 18, 24, and 30 Hz; common 45 Hz upper
  cutoff; weights `m^-1.25 + 0.25`, with one-based sub-band index `m`.
- Report overall accuracy, balanced accuracy, ITR, and participant-level
  values for every registered window.

These methods are references only. They are not architectural contributions
of HarmonicFoldNet and are not pooled with supervised cross-validation tests.

### Modern neural reference

- Model: the original MTSNet definition from upstream commit
  `890b0a4f93c3affd50741a1f1d036fb74a30a062`.
- Dataset: Dong2023.
- Windows: 0.4, 0.8, and 1.2 s.
- Evaluation: the existing grouped five-fold protocol, repeated at seeds 3407,
  12917, and 62003.
- Training schedule, optimizer, early stopping, preprocessing, and fold
  construction are inherited unchanged from the frozen external MTSNet runner.
- No architecture, hyperparameter, threshold, or preprocessing choice may be
  tuned after inspecting these extension results.

For inference, seed-level predictions are averaged within participant before
participant-level paired comparisons. The three planned HarmonicFoldNet-versus-
MTSNet contrasts form one Holm-corrected family.
Accuracy and balanced accuracy are both reported; accuracy is the primary
comparison metric for this extension.

## Efficiency accounting

For MTSNet and HarmonicFoldNet, report inference-graph parameter count,
MACs/FLOPs under the same convention, and batch-one single-thread CPU latency
on the same machine and input shape. Latency measurements must use identical
warm-up, repetition count, thread setting, and summary statistic. Training
time is not substituted for inference latency.

## Reporting boundary

All results from this protocol must be labeled supplementary/post-result.
Negative, neutral, and positive results are retained. The external datasets
remain independent tests of the already frozen model; this extension may
strengthen or narrow the interpretation, but it cannot retroactively redefine
the primary claim or model-selection process.

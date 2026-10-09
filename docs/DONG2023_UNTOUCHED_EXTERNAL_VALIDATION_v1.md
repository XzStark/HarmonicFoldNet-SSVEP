# Dong2023 untouched external-validation protocol v1

Frozen on 2026-10-09 before any HarmonicFoldNet or comparator accuracy was
computed on Dong2023. Public metadata and the source paper were inspected to
define the loader, latency and acquisition-domain interpretation. No model
output or label-versus-prediction comparison was inspected before freezing.

## Purpose

This is an independent cross-subject replication on a dataset that did not
participate in architecture design, hyperparameter selection, model retention
or manuscript claim selection. It is not a zero-shot weight-transfer test:
both architectures are trained on Dong2023 training participants and evaluated
on held-out Dong2023 participants.

Dong2023 must not be described as SNR-matched to Benchmark or BETA. The source
paper reports significantly lower fundamental SNR than the Tsinghua Benchmark.
It uses pre-gelled semi-dry electrodes, low-contrast grid stimuli and an
unshielded acquisition environment. The experiment therefore tests a second
practical low-SNR domain with a class grid aligned to Benchmark/BETA.

## Dataset contract

- Source: Dong and Tian, 2023, DOI 10.26599/BSA.2023.9050020.
- Stable mirror: https://zenodo.org/records/18847318.
- License: CC BY-NC 4.0; research use only; raw data are not redistributed.
- Independent units: 59 participants.
- Task: 40 JFPM targets, four blocks per participant.
- Frequencies: 8.0--15.8 Hz in 0.2-Hz increments.
- Phases: 0, 0.5 pi, pi and 1.5 pi repeated over target order.
- Channels: PO7, PO3, POz, PO4, PO8, O1, Oz and O2.
- Sampling: publisher-provided 250-Hz epoched data.
- Preprocessing: causal fourth-order 6--45-Hz SOS band-pass.
- Response onset: 0.16 s after stimulus onset, following the source paper's
  reported 159.49-ms mean visual latency.
- Post-stimulus samples are excluded.
- Registered windows: 0.4, 0.6, 0.8, 1.0, 1.2 and 1.5 s.

## Frozen models and splits

- Proposed model: retained HarmonicFoldNet v4.1; no Dong2023-specific model,
  optimizer, loss or threshold changes.
- Comparator: the existing protocol-adapted SSVEPformer implementation.
- Five deterministic participant-grouped outer folds; each participant appears
  in exactly one test fold.
- Within each outer fold, six non-test participants form the validation set;
  all other non-test participants form the training set.
- Split seed: 20260929.
- Training seeds: 20260929, 20260930 and 20260931.
- Checkpoint selection uses mean validation balanced accuracy over 0.4--1.2 s.
- Test participants are never used for stopping, selection or repair decisions.
- No architecture or training-protocol change is permitted after test results
  are inspected. Technical repairs require rerunning every affected job.

## Endpoints and analysis

- Primary endpoint: participant-level balanced accuracy at 1.2 s,
  HarmonicFoldNet versus SSVEPformer.
- Secondary windows: 0.4, 0.6, 0.8, 1.0 and 1.5 s.
- Accuracy and balanced accuracy are both recorded.
- Training seeds are averaged within participant before paired inference.
- Report paired participant differences, percentile-bootstrap 95% confidence
  intervals and two-sided Wilcoxon signed-rank tests.
- Holm correction is applied once across the six registered windows.
- Chance performance is 2.5% for 40 classes.

## Interpretation boundary

A favorable result supports generalization to an unseen, lower-SNR practical
acquisition domain with the same 40-class frequency grid. An unfavorable result
is retained as evidence against universal superiority. It must not be repaired
by tuning on Dong2023 and then called confirmatory external validation.

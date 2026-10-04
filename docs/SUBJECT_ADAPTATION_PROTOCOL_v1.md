# Participant-adaptation protocol (frozen before target evaluation)

## Purpose

Test whether a calibration-free HarmonicFold v4.1 model can use one or two labelled
participant blocks more efficiently than participant-trained TRCA and TDCA,
without changing the paper's zero-calibration primary endpoint.

## Leakage boundary

- Every participant is paired with the checkpoint from the outer fold in which
  that participant was a test participant. The participant is absent from both
  source training and source validation.
- For each held-out block, calibration uses only the immediately preceding one
  or two blocks (cyclic order). The held-out block is never used for fitting,
  epoch selection, or hyperparameter selection.
- Adaptation hyperparameters are selected on Kim2025 only. BETA is the untouched
  confirmatory dataset and is not used to revise the selected setting.
- One Benchmark participant was used for a three-epoch software smoke test
  before the selection grid. Benchmark adaptation results are therefore
  labelled secondary/exploratory rather than confirmatory.

## Adapter

The frozen v4.1 backbone receives an identity-initialized 8 x 8 spatial channel
matrix. Two candidates are considered during Kim2025 selection:

1. `spatial`: 64 participant-specific parameters.
2. `spatial_score`: the 64 spatial parameters plus the 49 parameters in the
   final shared candidate scorer (113 trainable parameters total).

No class-specific bias or target identity embedding is fitted. All BatchNorm
statistics and backbone Dropout states remain frozen.

## Hyperparameter selection

Selection set: the eight Kim2025 participants in outer fold 0, seed 20260929.

- Method: `spatial`, `spatial_score`
- Learning rate: 0.003, 0.01
- Epochs: 3, 10, 30
- Identity penalty: 0.01 (fixed)
- Adaptation schedule: 0.4, 0.8, and 1.2 second crops in deterministic shuffled
  order
- Selection score: mean participant balanced accuracy over 0.8 and 1.2 second
  test windows, averaged across the one-block and two-block budgets
- Tie break: fewer trainable parameters, then lower learning rate

After selection, the winning setting is frozen and run once on all BETA
participants, then on Benchmark as a secondary analysis. Zero-shot, one-block,
and two-block results are reported
alongside participant-calibrated TRCA and TDCA in a separate supervision table.

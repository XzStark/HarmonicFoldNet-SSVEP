# Subject-adaptation result v2

## Scope and correction

This result uses the frozen 113-parameter `spatial_score` adapter and the three
registered seeds (20260929--20260931) on all 70 public BETA participants. Each
participant is evaluated only by an outer-fold source checkpoint that excluded
that participant. Seed repeats are averaged within participant before paired
inference.

Version 1 called the classical comparator TRCA, but its filter construction was
ensemble TRCA (eTRCA): filters learned for every class were concatenated and
used to score every candidate. Version 2 implements and reports class-specific
TRCA and eTRCA separately. Neural-adapter results are unchanged.

## Adaptation result

Balanced accuracy (%):

| Calibration blocks | 0.4 s | 0.8 s | 1.2 s |
|---:|---:|---:|---:|
| 0 | 27.35 | 55.52 | 68.73 |
| 1 | 34.75 | 65.00 | 77.13 |
| 2 | 36.21 | 66.24 | 78.28 |

All six one-/two-block gains over zero calibration remain significant after
Holm correction. The adapter changes only 113 participant-specific parameters;
the source decoder stays frozen.

## Corrected two-block classical comparison

Balanced accuracy (%):

| Window | Adapter | TRCA | eTRCA | TDCA |
|---:|---:|---:|---:|---:|
| 0.4 s | 36.21 | 18.96 | 37.30 | 34.99 |
| 0.8 s | 66.24 | 34.60 | 55.77 | 56.55 |
| 1.2 s | 78.28 | 47.24 | 67.17 | 70.34 |

The adapter exceeds class-specific TRCA by 17.25, 31.64 and 31.04 percentage
points. It exceeds eTRCA by 10.47 points at 0.8 s and 11.11 points at 1.2 s,
and TDCA by 9.69 and 7.94 points at those windows. Those seven contrasts remain
significant after Holm correction over the complete corrected comparison
family. At 0.4 s, adapter versus eTRCA (-1.09 points) and TDCA (+1.22 points)
is not significant.

One-block TDCA is retained in machine-readable results as a low-sample
diagnostic, not a headline comparison. TRCA/eTRCA require at least two trials
per class in this implementation.

## Claim boundary

Supported: low-parameter participant adaptation yields repeatable gains and,
with two calibration blocks, outperforms TRCA across all registered windows and
outperforms eTRCA/TDCA at 0.8--1.2 s under the same BETA signal contract.

Not supported: shortest-window superiority over eTRCA, superiority without
target-participant calibration, transfer to the intended peri-auricular
hardware, or a clinical claim.

Machine-readable evidence:

- `runs/paper_v18/calibrated/beta_spatial_score_vs_trca_etrca_tdca.json`
- `runs/paper_v18/calibrated/beta_trca_etrca_tdca.json`
- `runs/paper_v15/adaptation_confirmatory/beta_spatial_score_3seed_summary.json`

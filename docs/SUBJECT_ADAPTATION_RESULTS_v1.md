# Frozen subject-adaptation result (v1)

> Superseded by `SUBJECT_ADAPTATION_RESULTS_v2.md`. The implementation labelled
> `TRCA` in this historical snapshot used the ensemble filter and is therefore
> eTRCA. The adapter values are unchanged; v2 separates TRCA and eTRCA and
> recomputes multiplicity correction across the corrected comparison family.

## Scope

This document freezes the first confirmatory result for the participant-specific
adapter defined in `SUBJECT_ADAPTATION_PROTOCOL_v1.md`. The architecture and
hyperparameters were selected only on the registered Kim2025 subset. BETA was
then evaluated without changing the 113-parameter `spatial_score` adapter,
learning rate, epoch count, identity penalty, crop schedule, or calibration
schedule.

The source model is `harmonic_fold_v4_1`. Every target participant is scored
only with an outer-fold checkpoint for which that participant was held out.
Results use seeds 20260929, 20260930 and 20260931. Participant scores are first
averaged across seeds; paired tests are then performed over the 70 independent
participants. The three seeds are not treated as additional participants.

## BETA result

Balanced accuracy (%):

| Calibration blocks | 0.4 s | 0.8 s | 1.2 s |
|---:|---:|---:|---:|
| 0 | 27.35 | 55.52 | 68.73 |
| 1 | 34.75 | 65.00 | 77.13 |
| 2 | 36.21 | 66.24 | 78.28 |

Standard deviation of the three seed-level means is 0.30--0.42 percentage
points for the adapted conditions. Relative to zero calibration, one block
improves the three windows by 7.40, 9.48 and 8.39 points; two blocks improve
them by 8.86, 10.72 and 9.55 points. All six participant-level paired tests
remain significant after Holm correction (`adjusted p <= 2.13e-12`).

## Protocol-matched classical comparison

With two calibration blocks, the frozen adapter records:

| Window | Adapter | TRCA | Adapter - TRCA | TDCA | Adapter - TDCA |
|---:|---:|---:|---:|---:|---:|
| 0.4 s | 36.21 | 37.30 | -1.09 | 34.99 | +1.22 |
| 0.8 s | 66.24 | 55.77 | +10.47 | 56.55 | +9.69 |
| 1.2 s | 78.28 | 67.17 | +11.11 | 70.34 | +7.94 |

The 0.8 s and 1.2 s differences are significant after Holm correction. The
0.4 s differences are not significant, so this result does not support a
shortest-window superiority claim. One-block TDCA is also reported in the raw
result, but is a weak low-sample reference; TRCA is undefined in this
implementation with only one training trial per class.

## Claim boundary

Supported: a 113-parameter identity-initialized adapter provides repeatable,
large participant-specific gains at 0.8--1.2 s and can outperform calibrated
TRCA/TDCA under the registered two-block BETA protocol.

Not yet supported: universal superiority at 0.4 s, superiority to every modern
deep decoder, transfer to the intended peri-auricular hardware, or clinical
effectiveness. Benchmark is a secondary exploratory replication; a licensed or
clean-room modern neural baseline and device measurements remain open gates.

Machine-readable evidence:

- `runs/paper_v15/adaptation_confirmatory/beta_spatial_score_3seed_summary.json`
- `runs/paper_v14/calibrated/beta_trca_tdca.json`

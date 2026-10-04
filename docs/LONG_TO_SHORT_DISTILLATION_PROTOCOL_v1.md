# Long-to-short distillation protocol v1

Status: independent BETA evaluation complete; rejected for the registered
short-window objective.

## Purpose

Test whether a long-window teacher can improve sub-second decoding while the
deployed student remains the unchanged HarmonicFold spectral v4.1 graph. This is a
training-only enhancement inspired by prior long-to-short SSVEP distillation;
it is not an architectural novelty claim.

## Development observation

One Benchmark development run (seed 20260929, grouped fold 0) was used only as
a go/no-go gate. Relative to the uniform v4.1 baseline, the frozen candidate
changed held-out balanced accuracy by +0.655, -0.179, +0.357, +0.893 and
+0.119 percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds respectively.

## Frozen intervention

- Teacher and student architecture: `harmonic_fold_v4_1`, full mode.
- Teacher training and model selection: 1.2-second windows only.
- Student training schedule: uniform 0.4, 0.6, 0.8, 1.0 and 1.2 seconds.
- Distillation applies only at 0.4 and 0.6 seconds.
- Loss: normal label-smoothed cross entropy plus 0.5 times KL divergence to
  the frozen 1.2-second teacher at temperature 2.0.
- The teacher uses the identical training/validation participant split as its
  paired student and never sees test labels during optimization or selection.
- Distillation is absent during inference; student parameters, operations and
  latency are identical to the v4.1 baseline.

No loss weight, temperature, window set or model component will be selected on
the BETA confirmatory results.

## Independent confirmation

Run three seeds and all five subject-disjoint grouped folds on BETA. Compare
paired participant predictions with the completed uniform-schedule v4.1 runs.
The primary endpoint is 0.4-second balanced accuracy. The candidate advances
only if the short-window gain generalizes and neither 0.8 nor 1.2 seconds shows
a material regression. All registered windows are reported with participant-
level paired tests, confidence intervals and Holm correction.

## Confirmatory result

The full three-seed, five-fold BETA run covered all 70 held-out participants:

| Window | Distilled | Baseline | Difference | Holm-adjusted p |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 s | 27.414% | 27.366% | +0.048 pp | 0.6985 |
| 0.6 s | 43.646% | 43.473% | +0.173 pp | 0.2518 |
| 0.8 s | 56.158% | 55.601% | +0.557 pp | 0.0340 |
| 1.0 s | 63.815% | 63.560% | +0.256 pp | 0.5295 |
| 1.2 s | 69.607% | 68.702% | +0.905 pp | 0.00011 |
| 1.5 s | 74.310% | 73.119% | +1.190 pp | 0.00000024 |

The registered 0.4-second endpoint did not improve meaningfully. The method is
therefore rejected as a short-window solution and does not replace v4.1.
The significant 0.8-, 1.2- and 1.5-second gains are retained as secondary
observations only; the paper must not redefine its primary claim after seeing
them. Machine-readable comparison:
`runs/paper_v26/long_to_short_formal/beta/comparison.json`.

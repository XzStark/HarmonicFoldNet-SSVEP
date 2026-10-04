# Short-window schedule protocol v1

Status: confirmatory Benchmark evaluation complete; candidate rejected.

## Purpose

This protocol tests whether the frozen HarmonicFold spectral v4.1 architecture can
improve sub-second decoding without adding inference parameters, changing the
deployment graph, or introducing a new architectural novelty claim.

## Frozen intervention

- Architecture: `harmonic_fold_v4_1`, `full` mode.
- Baseline training windows per schedule cycle: 0.4, 0.6, 0.8, 1.0, 1.2 s.
- Candidate schedule: 0.4, 0.4, 0.6, 0.8, 1.0, 1.2 s.
- Validation selection remains the unweighted mean over 0.4, 0.6, 0.8, 1.0
  and 1.2 s. Only the training exposure changes.
- Model width, local stages, reduced-token attention, optimizer, labels,
  participant splits and all preprocessing remain unchanged.

The extra 0.4-s exposure is a training-protocol optimization, not a claimed
architectural novelty. It adds no deployment parameters or inference cost.

## Development decision

The candidate was selected once on Benchmark five-fold split 0, seed 20260929.
The held-out participant accuracies were:

| Schedule | 0.4 s | 0.6 s | 0.8 s | 1.0 s | 1.2 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform baseline | 29.940% | 48.452% | 67.321% | 75.476% | 79.286% |
| Frozen 0.4-s x2 candidate | 30.714% | 49.167% | 67.976% | 77.202% | 79.643% |

No further schedule variants will be selected on this split.

## Confirmatory evaluation

Run the frozen candidate for seeds 20260929, 20260930 and 20260931 over all
five subject-disjoint folds on Benchmark and BETA. Compare paired participant
predictions against the already completed uniform-schedule v4.1 results at
every registered window. Report confidence intervals, paired tests and Holm
correction. The candidate is accepted only if the 0.4-s gain generalizes and
there is no material regression at 0.8 or 1.2 s.

## Confirmatory result and decision

The complete Benchmark three-seed, five-fold comparison produced:

| Window | Candidate | Uniform baseline | Difference | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| 0.4 s | 30.060% | 29.405% | +0.655 pp | 0.742 |
| 0.6 s | 48.817% | 48.770% | +0.048 pp | 1.000 |
| 0.8 s | 66.464% | 66.734% | -0.270 pp | 1.000 |
| 1.0 s | 75.310% | 75.583% | -0.274 pp | 1.000 |
| 1.2 s | 78.611% | 78.310% | +0.302 pp | 0.884 |

The apparent development-fold gain did not become a reliable confirmatory
effect. Because all differences are small and none survives multiplicity
correction, the candidate is rejected and the uniform schedule remains the
paper mainline. The planned BETA run was stopped after Benchmark confirmation
to avoid spending compute on an intervention that failed its acceptance gate.

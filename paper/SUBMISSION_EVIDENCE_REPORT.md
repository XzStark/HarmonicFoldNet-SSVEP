# Submission evidence restoration report

Generated from the preserved per-participant, per-seed run artifacts. The participant is the independent unit; the three training seeds are averaged within participant before inference.

## Restored evidence

- Main Benchmark and BETA results: 35 and 70 held-out participants, respectively, across seeds 20260929, 20260930 and 20260931.
- Participant-level ITR: calculated in each original run with 40 targets and selection time equal to the observation window plus 0.5 s, then averaged across seeds within participant.
- Final BETA component ablations: full, no-local, no-attention and no-harmonic-bias conditions for all 70 participants and three seeds.
- Attention mechanism analysis: four participant-level metrics across three windows with one conservative 12-test Holm family.
- MTSNet multiplicity audit: one Holm family across Benchmark/BETA and 0.4/0.8/1.2 s (six tests).

## Participant-level ITR

Values are mean [participant-bootstrap 95% CI] in bits/min. ITR is descriptive; the primary inferential endpoint remains participant-level balanced accuracy.

| Dataset | Window (s) | HarmonicFoldNet | Spectral Transformer | Mean difference [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Benchmark | 0.4 | 53.43 [39.54, 68.75] | 58.26 [43.75, 74.47] | -4.83 [-10.30, 1.09] |
| Benchmark | 0.6 | 94.87 [76.80, 114.61] | 86.97 [69.35, 105.54] | 7.90 [0.17, 15.99] |
| Benchmark | 0.8 | 128.40 [110.30, 146.13] | 116.90 [98.71, 135.13] | 11.50 [3.96, 19.22] |
| Benchmark | 1.0 | 134.50 [118.49, 149.54] | 122.87 [106.21, 138.75] | 11.63 [4.39, 18.66] |
| Benchmark | 1.2 | 125.58 [111.59, 138.75] | 114.80 [100.48, 128.51] | 10.78 [3.95, 17.10] |
| BETA | 0.4 | 47.29 [38.71, 55.97] | 55.30 [45.56, 65.49] | -8.00 [-11.70, -4.00] |
| BETA | 0.6 | 79.70 [67.75, 91.89] | 80.20 [67.49, 93.60] | -0.50 [-5.10, 4.07] |
| BETA | 0.8 | 98.20 [85.58, 111.04] | 91.00 [78.30, 104.05] | 7.20 [2.92, 11.52] |
| BETA | 1.0 | 104.03 [92.33, 115.76] | 92.19 [80.51, 103.95] | 11.85 [7.80, 16.04] |
| BETA | 1.2 | 103.51 [92.21, 114.36] | 88.44 [77.72, 99.12] | 15.07 [11.26, 19.04] |
| BETA | 1.5 | 96.72 [87.45, 105.78] | 80.87 [71.62, 89.84] | 15.85 [12.22, 19.61] |

## Final component ablations

Differences are full minus ablated balanced accuracy in percentage points. Holm correction is applied over the six registered windows separately for each prespecified component ablation.

| Ablation | Window (s) | Difference [95% CI], pp | Rank-biserial | Holm p |
| --- | ---: | ---: | ---: | ---: |
| no_local | 0.4 | 3.99 [3.21, 4.79] | 0.950 | 1.28e-11 |
| no_local | 0.6 | 5.36 [4.46, 6.27] | 0.945 | 1.28e-11 |
| no_local | 0.8 | 5.77 [4.95, 6.60] | 0.977 | 4.82e-12 |
| no_local | 1.0 | 6.62 [5.60, 7.65] | 0.976 | 4.82e-12 |
| no_local | 1.2 | 7.23 [6.27, 8.21] | 1.000 | 2.13e-12 |
| no_local | 1.5 | 7.19 [6.16, 8.28] | 0.999 | 2.13e-12 |
| no_attention | 0.4 | -0.66 [-1.32, 0.01] | -0.298 | 0.0330 |
| no_attention | 0.6 | 1.20 [0.46, 1.93] | 0.422 | 0.0074 |
| no_attention | 0.8 | 1.88 [1.15, 2.62] | 0.609 | 8.43e-05 |
| no_attention | 1.0 | 2.16 [1.32, 3.01] | 0.637 | 3.00e-05 |
| no_attention | 1.2 | 1.68 [0.80, 2.57] | 0.513 | 0.0008 |
| no_attention | 1.5 | 1.24 [0.21, 2.23] | 0.380 | 0.0122 |
| no_harmonic_bias | 0.4 | -0.85 [-1.58, -0.15] | -0.243 | 0.1553 |
| no_harmonic_bias | 0.6 | 0.43 [-0.29, 1.18] | 0.138 | 0.3222 |
| no_harmonic_bias | 0.8 | 1.04 [0.28, 1.84] | 0.303 | 0.0902 |
| no_harmonic_bias | 1.0 | 1.57 [0.68, 2.50] | 0.371 | 0.0346 |
| no_harmonic_bias | 1.2 | 1.11 [0.42, 1.87] | 0.327 | 0.0805 |
| no_harmonic_bias | 1.5 | 1.30 [0.56, 2.09] | 0.418 | 0.0163 |

## MTSNet multiplicity correction

| Dataset | Window (s) | Difference, pp | Raw p | Holm p (six-test family) |
| --- | ---: | ---: | ---: | ---: |
| Benchmark | 0.4 | -7.39 | 2.81e-06 | 1.40e-05 |
| Benchmark | 0.8 | 0.71 | 0.7681 | 1.0000 |
| Benchmark | 1.2 | 0.76 | 0.2415 | 0.9661 |
| BETA | 0.4 | -8.54 | 2.32e-12 | 1.39e-11 |
| BETA | 0.8 | -1.10 | 0.3536 | 1.0000 |
| BETA | 1.2 | 0.15 | 0.6885 | 1.0000 |

## Integrity result

All configured groups contained the registered three seeds and the complete participant set. Recomputed seed-averaged balanced-accuracy means matched the frozen comparison artifacts to numerical tolerance. No group-mean accuracy was substituted into the nonlinear ITR formula.

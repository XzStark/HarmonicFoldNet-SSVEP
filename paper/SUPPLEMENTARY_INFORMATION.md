# Supplementary information for HarmonicFoldNet

This supplement accompanies the manuscript *HarmonicFoldNet: A Compact Foldable Local-to-Global Decoder for Cross-Subject SSVEP Recognition*. It is generated from preserved run artifacts rather than transcribed from figures. The participant is the independent statistical unit throughout.

## Supplementary Methods S1. Evidence reconstruction and validation

The submission evidence builder reads the original per-fold result files for three training seeds. It validates the expected participant set, averages seed repeats within participant, reconstructs descriptive summaries and paired tests, and compares the reconstructed means with the frozen comparison records. Confidence intervals use 10,000 participant-bootstrap resamples. Paired tests use the two-sided Wilcoxon signed-rank test; effect sizes are rank-biserial correlations. Holm correction follows the multiplicity families declared in the manuscript and evidence configuration.

Individual audit trails are provided in `main_participant_metrics_by_seed.csv`, `main_participant_metrics_seed_averaged.csv`, and `ablation_participant_balanced_accuracy.csv`. The machine-readable configuration is `SUBMISSION_EVIDENCE_CONFIG.json`; the consolidated record is `submission_evidence.json`.

## Supplementary Methods S2. Registered dataset contracts

| Dataset | Participants | Classes | Role in this study |
| --- | --- | --- | --- |
| Kim2025 beta-range | 40 | 40 | Initial architecture development and beta-band stress testing |
| Tsinghua Benchmark | 35 | 40 | Subject-disjoint evaluation and later candidate-screening audit; not an untouched confirmatory dataset |
| BETA | 70 | 40 | Subject-disjoint evaluation, later candidate screening, component analysis, and optional participant adaptation; not untouched |
| Wearable SSVEP | 102 | 12 | Frozen directional wet/dry electrode-condition transfer after architecture selection |

Wet and dry Wearable recordings came from the same participants and therefore count once, giving 247 unique participants across the four datasets. Benchmark and BETA were later used to accept or reject follow-up variants while retaining the reported architecture; their intervals and tests are therefore selection-aware descriptive evidence rather than untouched confirmatory inference. Wearable remained the external electrode-condition evaluation.

| Dataset | Participants | Classes | Evaluated windows (s) | Primary figure windows (s) | Seeds | Outer folds |
| --- | --- | --- | --- | --- | --- | --- |
| TsinghuaBenchmark | 35 | 40 | 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0 | 0.4, 0.6, 0.8, 1.0, 1.2 | 3 | 5 |
| BETA | 70 | 40 | 0.4, 0.6, 0.8, 1.0, 1.2, 1.5 | 0.4, 0.6, 0.8, 1.0, 1.2, 1.5 | 3 | 5 |
| Wearable SSVEP | 102 | 12 | 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0 | 0.4, 0.6, 0.8, 1.0, 1.2 | 3 | 5 |

Benchmark windows of 2.0, 3.0 and 5.0 s were retained as a supplementary long-window audit and were included in the registered Benchmark multiplicity family. They were omitted from the primary figure to keep the wearable operating range legible. BETA used 0.4–1.5 s. The Wearable analysis registered 0.4–2.0 s; 0.4–1.2 s were the primary figure windows and 1.5/2.0 s were secondary transfer-window audits. All seven Wearable windows belong to the declared 21-test Holm family.

## Supplementary Table S1. Calibration-free balanced accuracy

Values are participant means with participant-bootstrap 95% confidence intervals. Seed repeats were averaged within participant before summarization.

| Dataset | Method | Window (s) | n | Balanced accuracy, % [95% CI] |
| --- | --- | --- | --- | --- |
| Benchmark | HarmonicFoldNet | 0.4 | 35 | 29.38 [24.44, 34.53] |
| Benchmark | HarmonicFoldNet | 0.6 | 35 | 48.77 [42.32, 55.31] |
| Benchmark | HarmonicFoldNet | 0.8 | 35 | 66.73 [60.13, 73.09] |
| Benchmark | HarmonicFoldNet | 1.0 | 35 | 75.60 [69.36, 81.27] |
| Benchmark | HarmonicFoldNet | 1.2 | 35 | 78.30 [72.13, 83.90] |
| Benchmark | HarmonicFoldNet | 2.0 | 35 | 85.31 [79.70, 90.27] |
| Benchmark | HarmonicFoldNet | 3.0 | 35 | 88.17 [82.78, 92.94] |
| Benchmark | HarmonicFoldNet | 5.0 | 35 | 87.16 [81.24, 91.99] |
| Benchmark | Spectral Transformer | 0.4 | 35 | 31.03 [25.78, 36.45] |
| Benchmark | Spectral Transformer | 0.6 | 35 | 46.10 [39.79, 52.56] |
| Benchmark | Spectral Transformer | 0.8 | 35 | 62.63 [55.53, 69.32] |
| Benchmark | Spectral Transformer | 1.0 | 35 | 71.20 [64.36, 77.37] |
| Benchmark | Spectral Transformer | 1.2 | 35 | 73.88 [67.31, 80.08] |
| Benchmark | Spectral Transformer | 2.0 | 35 | 78.28 [72.25, 83.88] |
| Benchmark | Spectral Transformer | 3.0 | 35 | 79.49 [73.65, 84.71] |
| Benchmark | Spectral Transformer | 5.0 | 35 | 79.42 [74.02, 84.32] |
| Beta | HarmonicFoldNet | 0.4 | 70 | 27.37 [24.19, 30.68] |
| Beta | HarmonicFoldNet | 0.6 | 70 | 43.47 [38.99, 47.96] |
| Beta | HarmonicFoldNet | 0.8 | 70 | 55.60 [50.66, 60.59] |
| Beta | HarmonicFoldNet | 1.0 | 70 | 63.56 [58.49, 68.43] |
| Beta | HarmonicFoldNet | 1.2 | 70 | 68.70 [63.69, 73.65] |
| Beta | HarmonicFoldNet | 1.5 | 70 | 73.12 [68.18, 77.78] |
| Beta | Spectral Transformer | 0.4 | 70 | 30.26 [26.82, 33.79] |
| Beta | Spectral Transformer | 0.6 | 70 | 43.46 [38.87, 48.12] |
| Beta | Spectral Transformer | 0.8 | 70 | 52.79 [47.82, 57.69] |
| Beta | Spectral Transformer | 1.0 | 70 | 58.52 [53.33, 63.53] |
| Beta | Spectral Transformer | 1.2 | 70 | 61.95 [56.93, 66.84] |
| Beta | Spectral Transformer | 1.5 | 70 | 65.09 [60.12, 70.06] |

## Supplementary Table S2. Paired main-model contrasts

| Dataset | Window (s) | n | HarmonicFoldNet minus reference, pp [95% CI] | Rank-biserial | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| Benchmark | 0.4 | 35 | -1.64 [-3.58, 0.48] | -0.362 | 0.0625 | 0.1088 |
| Benchmark | 0.6 | 35 | 2.68 [0.07, 5.42] | 0.378 | 0.0544 | 0.1088 |
| Benchmark | 0.8 | 35 | 4.11 [1.44, 6.74] | 0.526 | 0.0075 | 0.0224 |
| Benchmark | 1.0 | 35 | 4.40 [1.70, 7.02] | 0.587 | 0.0024 | 0.0098 |
| Benchmark | 1.2 | 35 | 4.42 [1.56, 7.22] | 0.614 | 0.0015 | 0.0076 |
| Benchmark | 2.0 | 35 | 7.03 [3.74, 10.17] | 0.697 | 0.0004 | 0.0027 |
| Benchmark | 3.0 | 35 | 8.68 [5.17, 12.08] | 0.765 | 2.37e-05 | 0.0002 |
| Benchmark | 5.0 | 35 | 7.74 [3.75, 11.44] | 0.693 | 0.0005 | 0.0031 |
| Beta | 0.4 | 70 | -2.90 [-4.15, -1.51] | -0.607 | 1.16e-05 | 3.47e-05 |
| Beta | 0.6 | 70 | 0.02 [-1.57, 1.59] | 0.004 | 0.9762 | 0.9762 |
| Beta | 0.8 | 70 | 2.82 [1.23, 4.44] | 0.446 | 0.0012 | 0.0023 |
| Beta | 1.0 | 70 | 5.04 [3.37, 6.70] | 0.685 | 8.98e-07 | 3.59e-06 |
| Beta | 1.2 | 70 | 6.75 [4.99, 8.62] | 0.808 | 4.14e-09 | 2.07e-08 |
| Beta | 1.5 | 70 | 8.03 [6.13, 9.98] | 0.866 | 2.98e-10 | 1.79e-09 |

## Supplementary Table S3. Participant-level information transfer rate

| Dataset | Window (s) | HarmonicFoldNet, bits/min [95% CI] | Spectral Transformer, bits/min [95% CI] |
| --- | --- | --- | --- |
| TsinghuaBenchmark | 0.4 | 53.43 [39.54, 68.75] | 58.26 [43.75, 74.47] |
| TsinghuaBenchmark | 0.6 | 94.87 [76.80, 114.61] | 86.97 [69.35, 105.54] |
| TsinghuaBenchmark | 0.8 | 128.40 [110.30, 146.13] | 116.90 [98.71, 135.13] |
| TsinghuaBenchmark | 1.0 | 134.50 [118.49, 149.54] | 122.87 [106.21, 138.75] |
| TsinghuaBenchmark | 1.2 | 125.58 [111.59, 138.75] | 114.80 [100.48, 128.51] |
| BETA | 0.4 | 47.29 [38.71, 55.97] | 55.30 [45.56, 65.49] |
| BETA | 0.6 | 79.70 [67.75, 91.89] | 80.20 [67.49, 93.60] |
| BETA | 0.8 | 98.20 [85.58, 111.04] | 91.00 [78.30, 104.05] |
| BETA | 1.0 | 104.03 [92.33, 115.76] | 92.19 [80.51, 103.95] |
| BETA | 1.2 | 103.51 [92.21, 114.36] | 88.44 [77.72, 99.12] |
| BETA | 1.5 | 96.72 [87.45, 105.78] | 80.87 [71.62, 89.84] |

## Supplementary Table S3b. ITR overhead sensitivity

Theoretical information transfer rate (ITR) was recomputed per participant and training seed for four assumed non-observation overheads. The table reports the window with the highest participant-mean ITR under each assumption; it shows that the optimal window is not invariant to the overhead model.

| Dataset | Method | Assumed overhead (s) | Peak window (s) | Peak mean ITR (bits/min) |
| --- | --- | --- | --- | --- |
| Benchmark | HarmonicFoldNet | 0.0 | 0.8 | 208.65 |
| Benchmark | HarmonicFoldNet | 0.25 | 1.0 | 161.40 |
| Benchmark | HarmonicFoldNet | 0.5 | 1.0 | 134.50 |
| Benchmark | HarmonicFoldNet | 1.0 | 1.0 | 100.87 |
| Benchmark | SSVEPformer | 0.0 | 0.8 | 189.97 |
| Benchmark | SSVEPformer | 0.25 | 1.0 | 147.44 |
| Benchmark | SSVEPformer | 0.5 | 1.0 | 122.87 |
| Benchmark | SSVEPformer | 1.0 | 1.0 | 92.15 |
| Beta | HarmonicFoldNet | 0.0 | 0.8 | 159.57 |
| Beta | HarmonicFoldNet | 0.25 | 1.0 | 124.84 |
| Beta | HarmonicFoldNet | 0.5 | 1.0 | 104.03 |
| Beta | HarmonicFoldNet | 1.0 | 1.2 | 79.99 |
| Beta | SSVEPformer | 0.0 | 0.8 | 147.88 |
| Beta | SSVEPformer | 0.25 | 0.8 | 112.67 |
| Beta | SSVEPformer | 0.5 | 1.0 | 92.19 |
| Beta | SSVEPformer | 1.0 | 1.0 | 69.14 |

## Supplementary Table S4. BETA component-ablation accuracy

| Condition | Window (s) | n | Balanced accuracy, % [95% CI] |
| --- | --- | --- | --- |
| full | 0.4 | 70 | 27.35 [24.09, 30.62] |
| full | 0.6 | 70 | 43.43 [38.98, 47.87] |
| full | 0.8 | 70 | 55.52 [50.44, 60.44] |
| full | 1.0 | 70 | 63.55 [58.58, 68.40] |
| full | 1.2 | 70 | 68.73 [63.74, 73.50] |
| full | 1.5 | 70 | 73.13 [68.21, 77.75] |
| no_local | 0.4 | 70 | 23.36 [20.68, 26.13] |
| no_local | 0.6 | 70 | 38.07 [33.99, 42.21] |
| no_local | 0.8 | 70 | 49.75 [44.88, 54.60] |
| no_local | 1.0 | 70 | 56.93 [51.91, 61.94] |
| no_local | 1.2 | 70 | 61.51 [56.14, 66.60] |
| no_local | 1.5 | 70 | 65.94 [60.79, 71.07] |
| no_attention | 0.4 | 70 | 28.01 [24.62, 31.32] |
| no_attention | 0.6 | 70 | 42.24 [37.66, 46.78] |
| no_attention | 0.8 | 70 | 53.64 [48.58, 58.65] |
| no_attention | 1.0 | 70 | 61.39 [56.20, 66.52] |
| no_attention | 1.2 | 70 | 67.05 [61.90, 72.02] |
| no_attention | 1.5 | 70 | 71.89 [66.95, 76.62] |
| no_harmonic_bias | 0.4 | 70 | 28.21 [24.83, 31.67] |
| no_harmonic_bias | 0.6 | 70 | 43.00 [38.40, 47.54] |
| no_harmonic_bias | 0.8 | 70 | 54.48 [49.33, 59.57] |
| no_harmonic_bias | 1.0 | 70 | 61.98 [56.72, 67.07] |
| no_harmonic_bias | 1.2 | 70 | 67.62 [62.36, 72.62] |
| no_harmonic_bias | 1.5 | 70 | 71.83 [66.74, 76.59] |

## Supplementary Table S5. BETA paired component effects

| Removed component | Window (s) | Full minus ablation, pp [95% CI] | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- |
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

## Supplementary Table S6. Harmonic-attention location audit

All rows use the same held-out checkpoints with the fixed bias present and masked. Confidence intervals are participant-bootstrap intervals for the paired difference. One Holm family covers all four metrics and three windows.

| Metric | Window (s) | Bias | Masked | Paired difference [95% CI] | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| Normalized entropy | 0.4 | 0.57 | 0.97 | -0.40 [-0.40, -0.40] | -1.000 | 4.27e-12 |
| Peak distance, Hz | 0.4 | 0.15 | 3.02 | -2.87 [-2.91, -2.83] | -1.000 | 4.27e-12 |
| Peak within ±0.5 Hz | 0.4 | 99.58 | 11.96 | 87.62 [87.11, 88.08] | 1.000 | 4.27e-12 |
| Harmonic-neighbourhood mass | 0.4 | 72.78 | 9.87 | 62.92 [62.86, 62.97] | 1.000 | 4.27e-12 |
| Normalized entropy | 0.8 | 0.56 | 0.97 | -0.41 [-0.41, -0.40] | -1.000 | 4.27e-12 |
| Peak distance, Hz | 0.8 | 0.17 | 2.81 | -2.64 [-2.70, -2.57] | -1.000 | 4.27e-12 |
| Peak within ±0.5 Hz | 0.8 | 98.92 | 17.93 | 81.00 [79.58, 82.33] | 1.000 | 4.27e-12 |
| Harmonic-neighbourhood mass | 0.8 | 73.53 | 10.22 | 63.30 [63.25, 63.35] | 1.000 | 4.27e-12 |
| Normalized entropy | 1.2 | 0.56 | 0.97 | -0.41 [-0.41, -0.41] | -1.000 | 4.27e-12 |
| Peak distance, Hz | 1.2 | 0.18 | 2.73 | -2.55 [-2.62, -2.47] | -1.000 | 4.27e-12 |
| Peak within ±0.5 Hz | 1.2 | 98.72 | 20.89 | 77.83 [75.95, 79.65] | 1.000 | 4.27e-12 |
| Harmonic-neighbourhood mass | 1.2 | 73.79 | 10.15 | 63.64 [63.55, 63.73] | 1.000 | 4.27e-12 |

## Supplementary Table S7. Calibration-free analytic references

Fixed harmonic power, CCA, and FBCCA were evaluated on the same causal 6–45 Hz, 250 Hz, eight-channel shards and all participants used by the neural comparisons. They require no participant-specific labels. Values are participant means with participant-bootstrap intervals; these descriptive controls were not included in an additional multiplicity family.

| Dataset | Method | Window (s) | n | Balanced accuracy, % [95% CI] |
| --- | --- | --- | --- | --- |
| Benchmark | Fixed harmonic power | 0.4 | 35 | 5.44 [4.74, 6.17] |
| Benchmark | Fixed harmonic power | 0.6 | 35 | 8.23 [7.26, 9.23] |
| Benchmark | Fixed harmonic power | 0.8 | 35 | 12.48 [10.67, 14.46] |
| Benchmark | Fixed harmonic power | 1.0 | 35 | 17.23 [14.33, 20.27] |
| Benchmark | Fixed harmonic power | 1.2 | 35 | 22.81 [19.34, 26.37] |
| Benchmark | CCA | 0.4 | 35 | 6.95 [6.05, 7.88] |
| Benchmark | CCA | 0.6 | 35 | 16.62 [13.86, 19.65] |
| Benchmark | CCA | 0.8 | 35 | 30.70 [25.86, 35.71] |
| Benchmark | CCA | 1.0 | 35 | 44.54 [38.25, 50.67] |
| Benchmark | CCA | 1.2 | 35 | 55.52 [48.76, 61.99] |
| Benchmark | FBCCA | 0.4 | 35 | 6.20 [5.52, 6.88] |
| Benchmark | FBCCA | 0.6 | 35 | 18.48 [15.42, 21.87] |
| Benchmark | FBCCA | 0.8 | 35 | 36.18 [30.87, 41.56] |
| Benchmark | FBCCA | 1.0 | 35 | 54.20 [47.67, 60.21] |
| Benchmark | FBCCA | 1.2 | 35 | 66.80 [59.69, 73.25] |
| BETA | Fixed harmonic power | 0.4 | 70 | 5.62 [5.04, 6.22] |
| BETA | Fixed harmonic power | 0.6 | 70 | 8.53 [7.63, 9.41] |
| BETA | Fixed harmonic power | 0.8 | 70 | 13.03 [11.55, 14.53] |
| BETA | Fixed harmonic power | 1.0 | 70 | 17.30 [15.28, 19.37] |
| BETA | Fixed harmonic power | 1.2 | 70 | 21.21 [18.81, 23.63] |
| BETA | Fixed harmonic power | 1.5 | 70 | 26.87 [23.79, 30.04] |
| BETA | CCA | 0.4 | 70 | 7.46 [6.55, 8.40] |
| BETA | CCA | 0.6 | 70 | 17.13 [14.84, 19.60] |
| BETA | CCA | 0.8 | 70 | 27.49 [23.58, 31.45] |
| BETA | CCA | 1.0 | 70 | 37.82 [32.73, 42.82] |
| BETA | CCA | 1.2 | 70 | 47.40 [41.86, 53.09] |
| BETA | CCA | 1.5 | 70 | 57.51 [51.52, 63.54] |
| BETA | FBCCA | 0.4 | 70 | 7.01 [6.19, 7.86] |
| BETA | FBCCA | 0.6 | 70 | 18.80 [16.30, 21.38] |
| BETA | FBCCA | 0.8 | 70 | 33.91 [29.99, 37.96] |
| BETA | FBCCA | 1.0 | 70 | 47.71 [42.89, 52.53] |
| BETA | FBCCA | 1.2 | 70 | 58.91 [53.71, 64.11] |
| BETA | FBCCA | 1.5 | 70 | 69.71 [64.46, 74.56] |

## Supplementary Table S8. Strong filter-bank Transformer comparison

The protocol-adapted filter-bank Transformer and HarmonicFoldNet used the same BETA participants, folds, windows and three training seeds. The six paired tests form one Holm family.

| Window (s) | HarmonicFoldNet, % [95% CI] | Filter-bank Transformer, % [95% CI] | Difference, pp [95% CI] | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 27.37 [24.17, 30.59] | 34.06 [30.45, 37.72] | -6.69 [-7.85, -5.53] | -0.939 | 7.35e-11 |
| 0.6 | 43.47 [39.03, 48.00] | 48.57 [43.95, 53.37] | -5.10 [-6.39, -3.82] | -0.821 | 1.53e-08 |
| 0.8 | 55.60 [50.66, 60.58] | 57.89 [52.88, 62.77] | -2.29 [-3.51, -1.12] | -0.426 | 0.0064 |
| 1.0 | 63.56 [58.55, 68.48] | 63.78 [58.65, 68.61] | -0.22 [-1.55, 1.05] | 0.000 | 0.9976 |
| 1.2 | 68.70 [63.62, 73.65] | 67.30 [62.21, 72.02] | 1.40 [-0.02, 2.71] | 0.366 | 0.0174 |
| 1.5 | 73.12 [68.29, 77.83] | 70.78 [65.98, 75.37] | 2.34 [0.84, 3.82] | 0.475 | 0.0022 |

## Supplementary Table S9. MTSNet six-test multiplicity family

The six registered Benchmark/BETA contrasts form one Holm family. This prevents a separate per-file correction from understating multiplicity.

| Dataset | Window (s) | n | Difference, pp [95% CI] | Rank-biserial | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| Benchmark | 0.4 | 35 | -7.39 [-9.52, -5.20] | -0.908 | 2.81e-06 | 1.40e-05 |
| Benchmark | 0.8 | 35 | 0.71 [-1.67, 3.10] | 0.057 | 0.7681 | 1.0000 |
| Benchmark | 1.2 | 35 | 0.76 [-1.76, 3.11] | 0.230 | 0.2415 | 0.9661 |
| Beta | 0.4 | 70 | -8.54 [-9.80, -7.23] | -0.965 | 2.32e-12 | 1.39e-11 |
| Beta | 0.8 | 70 | -1.10 [-2.49, 0.21] | -0.128 | 0.3536 | 1.0000 |
| Beta | 1.2 | 70 | 0.15 [-1.21, 1.50] | 0.055 | 0.6885 | 1.0000 |

## Supplementary Table S10. Optional participant adaptation

This is a retrospective cyclic leave-one-block evaluation. For each held-out block, the requested immediately preceding block or blocks were used for calibration, wrapping around at the first block; every block was predicted once. The neural adapter started from a source-trained HarmonicFoldNet checkpoint whose outer-fold training and validation participants excluded the target participant. TRCA, ensemble TRCA, and TDCA were fitted only on the target-user calibration blocks. The comparison therefore equalizes the amount of target-labelled calibration data, not total pretraining supervision, and should not be interpreted as prospective online adaptation.

TRCA and ensemble TRCA used one spatial component and class-average templates. Ensemble TRCA concatenated the class-specific filters for each class score. TDCA used four harmonics, five delay samples, one spatial component, class-specific sinusoidal-reference projections, and regularized generalized eigendecomposition. All calibrated methods used the same causal 6–45 Hz, 250 Hz, eight-channel shards.

| Condition | Calibration blocks | Window (s) | n | Balanced accuracy, % [95% CI] |
| --- | --- | --- | --- | --- |
| zero_shot | 0 | 0.4 | 70 | 27.35 [24.12, 30.54] |
| zero_shot | 0 | 0.8 | 70 | 55.52 [50.50, 60.35] |
| zero_shot | 0 | 1.2 | 70 | 68.73 [63.69, 73.68] |
| adapter_1_block | 1 | 0.4 | 70 | 34.75 [31.01, 38.52] |
| adapter_1_block | 1 | 0.8 | 70 | 65.00 [60.28, 69.46] |
| adapter_1_block | 1 | 1.2 | 70 | 77.13 [72.95, 81.07] |
| adapter_2_blocks | 2 | 0.4 | 70 | 36.21 [32.47, 40.01] |
| adapter_2_blocks | 2 | 0.8 | 70 | 66.24 [61.74, 70.64] |
| adapter_2_blocks | 2 | 1.2 | 70 | 78.28 [74.23, 82.11] |
| trca_2_blocks | 2 | 0.4 | 70 | 18.96 [15.98, 22.07] |
| trca_2_blocks | 2 | 0.8 | 70 | 34.60 [29.40, 39.96] |
| trca_2_blocks | 2 | 1.2 | 70 | 47.24 [41.22, 53.37] |
| etrca_2_blocks | 2 | 0.4 | 70 | 37.30 [32.32, 42.46] |
| etrca_2_blocks | 2 | 0.8 | 70 | 55.77 [49.50, 61.71] |
| etrca_2_blocks | 2 | 1.2 | 70 | 67.17 [60.90, 73.07] |
| tdca_2_blocks | 2 | 0.4 | 70 | 34.99 [30.64, 39.48] |
| tdca_2_blocks | 2 | 0.8 | 70 | 56.55 [51.22, 61.71] |
| tdca_2_blocks | 2 | 1.2 | 70 | 70.34 [65.21, 75.23] |

## Supplementary Table S11. Paired adaptation effects

TRCA and ensemble TRCA required two calibration blocks in this implementation. TDCA was also evaluated with one block as a diagnostic, yielding a 12-test adapter-versus-classical Holm family; the nine two-block comparisons are shown below, while the complete machine-readable table includes the three one-block TDCA contrasts.

| Left | Right | Window (s) | Paired difference, pp [95% CI] | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- | --- |
| adapter_1_block | zero_shot | 0.4 | 7.40 [6.35, 8.51] | 0.989 | 2.13e-12 |
| adapter_1_block | zero_shot | 0.8 | 9.48 [8.01, 10.99] | 1.000 | 2.13e-12 |
| adapter_1_block | zero_shot | 1.2 | 8.39 [6.78, 10.20] | 1.000 | 2.13e-12 |
| adapter_2_blocks | zero_shot | 0.4 | 8.86 [7.76, 9.98] | 1.000 | 2.13e-12 |
| adapter_2_blocks | zero_shot | 0.8 | 10.72 [9.24, 12.28] | 1.000 | 2.13e-12 |
| adapter_2_blocks | zero_shot | 1.2 | 9.55 [7.85, 11.48] | 1.000 | 2.13e-12 |
| adapter_2_blocks | trca_2_blocks | 0.4 | 17.25 [14.73, 19.84] | 0.986 | 5.18e-12 |
| adapter_2_blocks | trca_2_blocks | 0.8 | 31.64 [28.60, 34.64] | 0.998 | 4.27e-12 |
| adapter_2_blocks | trca_2_blocks | 1.2 | 31.04 [27.40, 34.68] | 0.997 | 4.27e-12 |
| adapter_2_blocks | etrca_2_blocks | 0.4 | -1.09 [-4.33, 1.99] | 0.001 | 0.9930 |
| adapter_2_blocks | etrca_2_blocks | 0.8 | 10.47 [7.03, 13.64] | 0.738 | 4.08e-07 |
| adapter_2_blocks | etrca_2_blocks | 1.2 | 11.11 [7.86, 14.40] | 0.829 | 1.01e-08 |
| adapter_2_blocks | tdca_2_blocks | 0.4 | 1.22 [-1.81, 4.18] | 0.172 | 0.4209 |
| adapter_2_blocks | tdca_2_blocks | 0.8 | 9.69 [6.69, 12.63] | 0.698 | 1.17e-06 |
| adapter_2_blocks | tdca_2_blocks | 1.2 | 7.94 [5.27, 10.62] | 0.723 | 7.29e-07 |

## Supplementary Table S12. Directional electrode-condition transfer

| Train to test | Window (s) | Balanced accuracy, % [95% CI] |
| --- | --- | --- |
| dry_to_dry | 0.4 | 27.63 [24.94, 30.56] |
| dry_to_dry | 0.6 | 36.40 [32.90, 39.91] |
| dry_to_dry | 0.8 | 44.73 [40.67, 48.89] |
| dry_to_dry | 1.0 | 51.16 [46.86, 55.52] |
| dry_to_dry | 1.2 | 55.05 [50.46, 59.65] |
| dry_to_dry | 1.5 | 57.44 [52.70, 61.97] |
| dry_to_dry | 2.0 | 60.03 [55.35, 64.63] |
| dry_to_wet | 0.4 | 35.65 [32.39, 39.06] |
| dry_to_wet | 0.6 | 46.35 [42.25, 50.35] |
| dry_to_wet | 0.8 | 54.58 [50.11, 58.98] |
| dry_to_wet | 1.0 | 59.85 [55.24, 64.51] |
| dry_to_wet | 1.2 | 64.25 [59.61, 68.87] |
| dry_to_wet | 1.5 | 66.56 [61.98, 71.15] |
| dry_to_wet | 2.0 | 68.32 [63.60, 72.96] |
| wet_to_dry | 0.4 | 24.00 [21.02, 27.10] |
| wet_to_dry | 0.6 | 30.48 [26.63, 34.40] |
| wet_to_dry | 0.8 | 35.30 [30.80, 39.90] |
| wet_to_dry | 1.0 | 38.97 [34.03, 43.98] |
| wet_to_dry | 1.2 | 42.29 [37.26, 47.56] |
| wet_to_dry | 1.5 | 44.99 [39.55, 50.56] |
| wet_to_dry | 2.0 | 47.78 [42.18, 53.42] |
| wet_to_wet | 0.4 | 45.19 [41.60, 48.94] |
| wet_to_wet | 0.6 | 56.45 [52.16, 60.58] |
| wet_to_wet | 0.8 | 64.30 [59.91, 68.58] |
| wet_to_wet | 1.0 | 69.06 [64.76, 73.37] |
| wet_to_wet | 1.2 | 72.64 [68.38, 76.80] |
| wet_to_wet | 1.5 | 74.79 [70.54, 78.92] |
| wet_to_wet | 2.0 | 77.43 [73.33, 81.40] |

## Supplementary Table S13. Paired electrode-transfer effects

| Contrast | Window (s) | Difference, pp [95% CI] | Rank-biserial | Holm p |
| --- | --- | --- | --- | --- |
| dry_to_wet minus wet_to_wet | 0.4 | -9.55 [-11.14, -7.95] | -0.890 | 1.24e-13 |
| dry_to_wet minus wet_to_wet | 0.6 | -10.10 [-11.84, -8.29] | -0.878 | 2.84e-13 |
| dry_to_wet minus wet_to_wet | 0.8 | -9.72 [-11.76, -7.73] | -0.851 | 2.04e-12 |
| dry_to_wet minus wet_to_wet | 1.0 | -9.22 [-11.31, -7.21] | -0.845 | 4.83e-12 |
| dry_to_wet minus wet_to_wet | 1.2 | -8.39 [-10.44, -6.39] | -0.827 | 1.12e-11 |
| dry_to_wet minus wet_to_wet | 1.5 | -8.23 [-10.37, -6.15] | -0.806 | 3.31e-11 |
| dry_to_wet minus wet_to_wet | 2.0 | -9.11 [-11.16, -7.16] | -0.874 | 5.99e-13 |
| wet_to_dry minus dry_to_dry | 0.4 | -3.63 [-5.70, -1.71] | -0.352 | 0.0021 |
| wet_to_dry minus dry_to_dry | 0.6 | -5.92 [-8.64, -3.36] | -0.417 | 0.0005 |
| wet_to_dry minus dry_to_dry | 0.8 | -9.44 [-12.75, -6.25] | -0.566 | 3.70e-06 |
| wet_to_dry minus dry_to_dry | 1.0 | -12.19 [-16.17, -8.46] | -0.627 | 3.07e-07 |
| wet_to_dry minus dry_to_dry | 1.2 | -12.76 [-17.07, -8.73] | -0.591 | 1.31e-06 |
| wet_to_dry minus dry_to_dry | 1.5 | -12.45 [-16.80, -8.23] | -0.565 | 3.70e-06 |
| wet_to_dry minus dry_to_dry | 2.0 | -12.25 [-17.05, -7.76] | -0.482 | 7.69e-05 |
| dry_to_wet minus wet_to_dry | 0.4 | 11.64 [8.35, 15.09] | 0.703 | 7.03e-09 |
| dry_to_wet minus wet_to_dry | 0.6 | 15.86 [11.75, 20.17] | 0.758 | 2.97e-10 |
| dry_to_wet minus wet_to_dry | 0.8 | 19.29 [14.54, 24.20] | 0.794 | 4.13e-11 |
| dry_to_wet minus wet_to_dry | 1.0 | 20.87 [15.82, 25.99] | 0.804 | 2.44e-11 |
| dry_to_wet minus wet_to_dry | 1.2 | 21.96 [16.81, 27.27] | 0.806 | 2.40e-11 |
| dry_to_wet minus wet_to_dry | 1.5 | 21.57 [16.25, 27.13] | 0.791 | 4.52e-11 |
| dry_to_wet minus wet_to_dry | 2.0 | 20.54 [15.16, 26.07] | 0.740 | 7.84e-10 |

## Supplementary Table S14. Deployment measurements

| Method | Window (s) | Parameters | CPU median, ms | CPU p95, ms | GPU median, ms | GPU p95, ms | Peak GPU, MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FB-SSVEPformer | 0.4 | 3,741,724 | 2.756 | 3.593 | 7.601 | 9.602 | 23.57 |
| FB-SSVEPformer | 0.8 | 3,741,724 | 2.750 | 4.134 | 7.743 | 11.764 | 23.58 |
| FB-SSVEPformer | 1.2 | 3,741,724 | 2.673 | 4.270 | 7.753 | 11.734 | 23.58 |
| HarmonicFoldNet | 0.4 | 435,043 | 2.991 | 4.008 | 3.465 | 4.506 | 12.09 |
| HarmonicFoldNet | 0.8 | 435,043 | 2.899 | 4.068 | 3.717 | 5.357 | 13.26 |
| HarmonicFoldNet | 1.2 | 435,043 | 3.040 | 4.390 | 3.428 | 5.522 | 14.42 |
| MTSNet reconstruction | 0.4 | 8,718,800 | 5.512 | 7.159 | 3.179 | 4.969 | 43.26 |
| MTSNet reconstruction | 0.8 | 10,484,200 | 6.889 | 8.309 | 2.888 | 4.064 | 49.63 |
| MTSNet reconstruction | 1.2 | 12,569,600 | 7.588 | 9.409 | 2.801 | 3.649 | 57.60 |
| SSVEPformer | 0.4 | 1,247,240 | 0.874 | 1.173 | 2.611 | 3.258 | 14.03 |
| SSVEPformer | 0.8 | 1,247,240 | 0.933 | 1.178 | 2.617 | 3.890 | 14.03 |
| SSVEPformer | 1.2 | 1,247,240 | 0.826 | 1.149 | 2.528 | 3.656 | 14.04 |

These are empirical within-session quantiles from 1,000 batch-one float32 passes after 100 warm-up passes on one laptop, with one timed CPU thread. They do not estimate between-session, power-state, or thermal-state uncertainty. Exact source JSON hashes and the benchmarked checkpoint hash are recorded in the source-data package.

The fold-equivalence audit covered 20 preserved checkpoints: all 15 BETA fold-seed checkpoints and five Benchmark folds for seed 20260929. At five registered windows, the first trial of every class for every held-out participant yielded 49,000 paired predictions. The folded and training graphs produced zero label disagreements; the maximum absolute logit error was 1.55 × 10^-5. Full checkpoint-window rows are stored in `folding_equivalence.json`.

## Supplementary Table S15. Comparator implementation provenance

| Comparator | Source anchor | Local protocol adaptation | Redistribution |
| --- | --- | --- | --- |
| SSVEPformer / FB-SSVEPformer | Published architecture; public reproduction commit fa21513054c8 | Eight channels; 40 classes; 250 Hz; 0.25 Hz spectral grid; common 6–45 Hz band; three FB branches at 6/14/22 Hz | Local clean implementation; no third-party source copied |
| MTSNet reconstruction | Upstream model definition commit 890b0a4f93c3 | Common participant folds, windows, optimizer budget, validation selection, channel contract, and one duration-specific model | Upstream source is not redistributed |
| CCA / FBCCA / harmonic power | Declared analytic definitions in the manuscript and release code | Same 250 Hz, eight-channel, 6–45 Hz evaluation shards; no participant labels | Included in the source release |

Published values obtained under different channel, class, calibration, segmentation, or split contracts were not inserted into the protocol-matched performance tables. DGConformer, SED-xLSTM, and other recent systems remain literature references because a matched local reconstruction was not completed.

## Supplementary Methods S3. Development record and negative results

The final architecture was selected after a bounded development sequence. Tested alternatives included increased width, alternative local domains and receptive fields, nested sub-bands, reliability gates, duration conditioning, class-grid alignment, candidate-local mixing, short-window resampling, long-to-short distillation, temporal segment changes, and explicit phase dynamics. These experiments are documented in the repository and are not counted as independent confirmatory tests. They are reported to expose the search history and reduce selective reporting. The final evidence tables use the frozen v4.1 architecture only.

## Supplementary Methods S4. Reproducibility manifest

| File | Bytes | SHA-256 |
| --- | --- | --- |
| paper/SUBMISSION_EVIDENCE_CONFIG.json | 8437 | 6dbca720059e073087beac155abf82cffcd0880b5837e53ad18826baadafd12c |
| paper/source_data/submission_evidence.json | 256107 | 26373956d5a4d2636d51e70b420e8d0c348bab3e860afa96f9704df519a760ad |
| paper/source_data/main_participant_metrics_by_seed.csv | 317509 | 038adf0cc61b72f53f9f9d8d37b3c02b16635d94a7fb7acc942dd10465393a86 |
| paper/source_data/main_participant_metrics_seed_averaged.csv | 116590 | ba7c7c6290d01261f5d0f4ab156b29b411a1d4c9526a2ae1bea82c06ee82ed74 |
| paper/source_data/itr_overhead_sensitivity.csv | 7839 | 7cf09c0ed9b322d1233b1214598cd567b064ef6b9dbf3cf794b62de69fc9f888 |
| paper/source_data/itr_overhead_sensitivity_peaks.csv | 847 | 1085933212624b71108861ee4f60278500fc0bbd390738015e24427967fab1b6 |
| paper/source_data/ablation_participant_balanced_accuracy.csv | 72027 | 52eaa98af7a853001ba1f7a9d225d431949eb7480cb7f5386ccce3884c191184 |
| paper/source_data/attention_mechanism_participant_metrics.csv | 64088 | fb3af08fa047a58ff48a68f7224f379b062def6812c01234d5e7adbc64d129ad |
| paper/source_data/attention_mechanism_summary.csv | 3682 | 2ca278ccaf60ca84c89693a2c7e0dfb878f7990ac8fdfb0441d93bfd8046fb4d |
| paper/source_data/analytic_baseline_participant.csv | 69989 | 53ea6f20c672fd58595cb9b7a8d51526313d12314643e20864059807e08764e5 |
| paper/source_data/analytic_baseline_summary.csv | 6758 | 2079717ec98c914d5d1e55f2a45cf4f786f8d999d84292ca39ba5d5c03bf24f9 |
| paper/source_data/strong_filter_bank_participant_by_seed.csv | 109502 | 9a5a82ed23ada6c913076c90f25106147f5163173da00e4d78a06314e7cd6821 |
| paper/source_data/strong_filter_bank_participant_seed_averaged.csv | 38995 | 911a90d44a092dc6ef110c87699f6b50566a79fea725c3fcaa5953afae5bb619 |
| paper/source_data/strong_filter_bank_summary.csv | 2215 | e1f20c01cb4119dc626fcdbdcc1c545196016a87d66ab260a085c96e5a3c9e3b |
| paper/source_data/strong_filter_bank_contrasts.csv | 1746 | d0e82fa62d455a24839538e10607d7e4c6244bb54c6bb60dcc03cf79f5bfc684 |
| paper/source_data/adaptation_participant_balanced_accuracy.csv | 112873 | 9723f0c62e3c6ee2f02616d29dc008a4681246a1793344f3a9fd1852f6528239 |
| paper/source_data/adaptation_summary.csv | 3052 | 6e63af75ad60865b40c3e459c71c2df48d3885805c1c633fb3d1ce2e341c1819 |
| paper/source_data/adaptation_contrasts.csv | 4180 | bf003ad87e4f7b577835f8d7d328e65b57f9ac9d970c0edfbb3ad3445f779a3a |
| paper/source_data/electrode_transfer_participant_by_seed.csv | 464436 | 72adab8b8d5fbdf1e096d37c7cb9ac5257d5561749bec8fc627f27f6c2eb1692 |
| paper/source_data/electrode_transfer_participant_seed_averaged.csv | 138959 | 511ba69da286c1df7dfa1a359c92a21775a77c534fbe6be73c4d754894ea6a34 |
| paper/source_data/electrode_transfer_summary.csv | 5072 | f579a2ac609869f02e95d55e69b8bf17e35b2cd7a613e013ae058b8ec2d2f84c |
| paper/source_data/electrode_transfer_contrasts.csv | 6012 | 2950e3b8209c9b5c359516dcc3e44cc868fe3d877f1fe6dd0779a7e72b80dd5c |
| paper/source_data/mtsnet_contrasts_corrected.csv | 2167 | a072f074a8c7cd1333c5fcbc84b63df2e48d4e438935f6e00c0576ad8886dbba |
| paper/source_data/mtsnet_participant_by_seed.csv | 82826 | eb2afa2282bf9675abc04ef4450eb24736a281e2f5b89ac77697f0984a338f60 |
| paper/source_data/mtsnet_participant_seed_averaged.csv | 28043 | bc4f2bfc4674e25103a7080d566ceabc2a300c35502522762c402dc7698a00fd |
| paper/source_data/mtsnet_summary.csv | 2191 | 2335e727d7ca30a41f7a09d5dfad7006457d7e26a9e4fc9a7ad5e5cb667802aa |
| paper/source_data/folding_equivalence.json | 37031 | e606e81549193124c868b9ded8c19b36ca7e92ae3e3572db25675309736e93b8 |
| paper/source_data/split_manifest.csv | 93598 | b6a189a0550e75baa6c5b511e4263ad461822c9310bcd06148cb31fda54fb360 |
| paper/source_data/split_manifest_audit.json | 366 | 2068af089bbbb5c741bb4befe535d0a492c121e39f1a0865be91af689b39c9eb |
| paper/source_data/deployment_measurements.csv | 6151 | dad81464c16e40ddf5853a60524ea350d01063e0a6f687a7cfc9df3425519acb |
| paper/source_data/deployment_checkpoint_manifest.csv | 192 | 370b520e436b5c46020121216c1af32460c68be148802fed01da2bb4fe21dea9 |

The hash values identify the exact source-data files used to generate this supplement. Regenerating the evidence should reproduce the numerical tables; bootstrap intervals are deterministic under seed 20261003.

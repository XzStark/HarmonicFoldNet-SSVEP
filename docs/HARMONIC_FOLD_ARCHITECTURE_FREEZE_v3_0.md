# HarmonicFold spectral architecture freeze v3.0

## Status

`spectral_fusion_v3_0` is frozen after development and grouped ablation on
Kim2025. Kim2025 remains development evidence only. BETA and Wearable are
reserved for external validation and must not be used to alter the architecture
or hyperparameters after their results are observed.

## Frozen model and protocol

- Fixed-grid complex FFT front end at 0.25 Hz resolution over 6--45 Hz.
- Segmented phase-demodulation candidate tokens as the stable candidate path.
- Complex harmonic-neighborhood candidate tokens fused through an
  identity-initialized residual projection.
- Two early foldable local-mixing stages on the frequency axis. Their residual
  gates are initialized at logit 0.0 (weight 0.5), rather than the v2.3 value
  of -2.0 (weight about 0.119).
- Frequency-token reduction before late self-attention and candidate
  cross-attention.
- Position-preserving lightweight global spectral classification residual.
- Analytic CCA evidence only as an optional residual prior.
- Width 48, local depths `[1, 1]`, one attention block, four heads, four
  harmonics, two spectral-neighborhood bins, and dropout 0.1.
- Training and model selection use the registered windows 0.4/0.6/0.8/1.0/
  1.2 s; longer windows are reported but are not tuning objectives.

The implementation is original 1-D EEG code. It applies the HarmonicFold design
principles of early local mixing, late global attention, hierarchical token
reduction, and deployment-time branch folding. It does not copy Apple source
code, weights, image operators, or outputs.

## Development evidence

On the fixed Kim2025 development split, v3.0 balanced accuracy for
0.4/0.6/0.8/1.0/1.2 s was 10.83/23.44/37.29/49.58/57.81%. This was not used
as external evidence.

In five participant-disjoint Kim2025 folds (40 paired participants, one fixed
seed), full v3.0 versus `no_local` produced the following mean balanced
accuracy differences:

| Window | Full | No local | Difference | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: |
| 0.4 s | 10.30% | 9.58% | +0.72 pp | 0.1693 |
| 0.6 s | 21.94% | 20.65% | +1.29 pp | 0.0168 |
| 0.8 s | 32.71% | 31.34% | +1.36 pp | 0.0014 |
| 1.0 s | 43.42% | 42.07% | +1.34 pp | 0.0157 |
| 1.2 s | 49.19% | 48.25% | +0.94 pp | 0.1696 |

The local branch therefore has reproducible short-window value at 0.6--1.0 s
under this development protocol. It is not claimed to improve all windows:
at 2/3/5 s the difference was approximately zero.

## Interpretation and claim boundary

The result supports retaining local-to-global local spectral mixing in the main
model. It does not prove superiority over all Transformer or calibration-based
methods, and it is not an external-validation result. The paper claim remains
conditional on fixed-model replication on BETA and Wearable, strong baselines,
multi-seed stability, and deployment measurements.

## External validation gate

From this freeze onward:

1. do not change architecture, dropout, width, depth, spectral grid, windows,
   optimizer, or selection rule after viewing BETA or Wearable results;
2. run participant-disjoint full/no-local and SSVEPformer comparisons first;
3. expand to all registered seeds only if the predeclared screening rule is
   met, without tuning on external results;
4. report paired participant-level uncertainty, effect sizes, and corrected
   significance rather than only aggregate accuracy;
5. keep calibration-free neural comparisons separate from calibrated
   TRCA/eTRCA/TRCA-R/TDCA comparisons;
6. complete no-attention/no-evidence ablations and folded-graph deployment
   equivalence before making the final paper claim.

Primary development statistic:
`runs/paper_v7/statistics/kim_v3_0_full_vs_no_local_seed20260929.json`.

# Final evidence summary v1

Date: 2026-10-03

## Final model

HarmonicFold spectral v4.1 is the frozen main model. It keeps the registered
local-to-global sequence:

1. reparameterizable local temporal mixing;
2. complex spectral and candidate-aligned temporal representations;
3. spectral-token reduction;
4. late harmonic-biased candidate-to-spectrum attention;
5. candidate attention and shared scoring;
6. train-to-deploy folding of the local mixer.

The model has 435,139 training-graph parameters and 435,043 folded-graph
parameters. The same weights accept all registered observation windows.

## Zero-target-data accuracy track

All values below are participant-level means after averaging three seeds over
complete grouped folds. P values are paired participant-level tests after Holm
correction.

### HarmonicFold v4.1 versus protocol-matched SSVEPformer

| Dataset | Window | HarmonicFold | SSVEPformer | Difference | Holm p |
|---|---:|---:|---:|---:|---:|
| Benchmark | 0.4 s | 29.38% | 31.03% | -1.64 pp | 0.1088 |
| Benchmark | 0.6 s | 48.77% | 46.10% | +2.68 pp | 0.1088 |
| Benchmark | 0.8 s | 66.73% | 62.63% | +4.11 pp | 0.0224 |
| Benchmark | 1.0 s | 75.60% | 71.20% | +4.40 pp | 0.0098 |
| Benchmark | 1.2 s | 78.30% | 73.88% | +4.42 pp | 0.0076 |
| BETA | 0.4 s | 27.37% | 30.26% | -2.90 pp | 3.47e-5 |
| BETA | 0.6 s | 43.47% | 43.46% | +0.02 pp | 0.9762 |
| BETA | 0.8 s | 55.60% | 52.79% | +2.82 pp | 0.0023 |
| BETA | 1.0 s | 63.56% | 58.52% | +5.04 pp | 3.59e-6 |
| BETA | 1.2 s | 68.70% | 61.95% | +6.75 pp | 2.07e-8 |
| BETA | 1.5 s | 73.12% | 65.09% | +8.03 pp | 1.79e-9 |

### Stronger modern references

Against protocol-adapted FB-SSVEPformer on BETA, v4.1 trails by 6.69, 5.10
and 2.29 points at 0.4, 0.6 and 0.8 seconds, is tied at 1.0 second, and leads
by 1.40 and 2.34 points at 1.2 and 1.5 seconds. All differences except 1.0
second are significant after correction.

Against the pinned MTSNet protocol reconstruction, v4.1 trails significantly
at 0.4 seconds on Benchmark and BETA. At 0.8 and 1.2 seconds, the differences
are not significant on either dataset: +0.71/+0.76 points on Benchmark and
-1.10/+0.15 points on BETA.

These results support a window-dependent Pareto, not universal accuracy
superiority.

## Calibrated user-adaptation track

The optional adapter changes only 113 parameters and is evaluated separately
because it uses target-user calibration data. On BETA, two calibration blocks
raise v4.1 from 27.35/55.52/68.73% to 36.21/66.24/78.28% at 0.4/0.8/1.2
seconds. It exceeds eTRCA and TDCA at 0.8 and 1.2 seconds under the recorded
matched calibration track; at 0.4 seconds it is statistically comparable, not
universally superior.

## Mechanism evidence

The harmonic bias moves more than 72% of late-attention mass into the target
harmonic neighbourhood, versus roughly 10% without the bias. The independent
no-bias ablation shows significant accuracy gains at 1.0 and 1.5 seconds, but
not a universal short-window gain. This separates an interpretable mechanism
claim from an unsupported claim that harmonic bias solves 0.4-second decoding.

## Deployment Pareto

On the same laptop, batch 1, float32, one CPU thread and 1,000 timed runs:

- folded v4.1 CPU P50 is 2.90--3.04 ms and CUDA P50 is 3.43--3.72 ms;
- ordinary SSVEPformer is faster: CPU P50 0.83--0.93 ms;
- v4.1 is faster on CPU than FB-SSVEPformer and MTSNet;
- v4.1 uses 12.09--14.42 MiB peak CUDA allocation, below FB-SSVEPformer and
  much below MTSNet;
- folding preserves logits within at most 1.43e-6 absolute error.

The deployable claim is compactness and a measured accuracy/size/latency/
memory Pareto. It is not a claim that HarmonicFold naming guarantees the lowest
latency on every device.

## Completed evidence lines

| Evidence line | Status |
|---|---|
| Strong, fair baselines | Complete: SSVEPformer, FB-SSVEPformer, MTSNet reconstruction and separate classical calibrated track |
| Mechanism evidence | Complete: attention mass, harmonic-bias ablation, local and attention-path ablations |
| Short-window and cross-participant generalization | Complete: complete grouped folds, three seeds, Benchmark and BETA; wearable dry/wet transfer recorded separately |
| Deployment Pareto | Complete: folded equivalence, parameters, CPU/CUDA P50/P95 and peak memory |

## Negative-results boundary

The search tested and rejected larger width, alternative local domains and
receptive fields, nested subbands, reliability gates, duration conditioning,
class-grid alignment, candidate-local mixing, short-window resampling,
long-to-short distillation, temporal segment counts and explicit phase
dynamics. The final phase-dynamics candidate reduced 0.4-second held-out
accuracy by 0.66 points and therefore closed the architecture-search cycle.

## Publication conclusion

The evidence is sufficient to begin a full manuscript for a lightweight BCI
or wearable neural-engineering venue. The strongest defensible contribution is
the complete architecture-and-evidence package: a compact universal-window
decoder, harmonic candidate conditioning with direct mechanism analysis, a
separate ultra-small personalization path, and transparent deployment Pareto.

The evidence is not sufficient for an all-window SOTA claim. The short-window
deficit is a reported limitation and a future-work boundary, not a result to be
hidden or tuned away on the same development folds.

## Primary artifacts

- `docs/FINAL_MODEL_DECISION_v1.md`
- `docs/DEPLOYMENT_PARETO_RESULTS_v1.md`
- `docs/HARMONIC_ATTENTION_MECHANISM_RESULTS_v1.md`
- `docs/SUBJECT_ADAPTATION_RESULTS_v2.md`
- `docs/SHORT_WINDOW_FRONTEND_DIAGNOSIS_PROTOCOL_v1.md`
- `runs/paper_v37/comparisons/benchmark_harmonic_fold_vs_ssvepformer_3seed.json`
- `runs/paper_v37/comparisons/beta_harmonic_fold_vs_ssvepformer_3seed.json`
- `runs/paper_v36/comparisons/beta_harmonic_fold_vs_fb_ssvepformer_3seed.json`
- `runs/paper_v17/mtsnet_formal/`
- `runs/paper_v39/deployment/`

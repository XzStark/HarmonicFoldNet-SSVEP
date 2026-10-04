# HarmonicFold spectral architecture freeze v2.3

## Status

`spectral_fusion_v2_3` is frozen after development on Kim2025. Kim2025 is
therefore development evidence only and must not be described as untouched
external validation.

## Frozen model

- Fixed-grid complex FFT front end at 0.25 Hz resolution over 6--45 Hz.
- Segmented phase-demodulation candidate tokens retained as the stable path.
- Complex harmonic-neighborhood candidate tokens fused through an
  identity-initialized residual projection.
- Early foldable local-mixing blocks on the frequency axis.
- Frequency-token reduction before late self-attention and candidate
  cross-attention.
- Position-preserving lightweight global spectral classification residual.
- Analytic CCA evidence used only as an optional residual prior.
- Width 48, local depths `[1, 1]`, one attention block, four heads, four
  harmonics, two spectral-neighborhood bins, and dropout 0.1.

The implementation is original 1-D EEG code. It uses the HarmonicFold design
principles of local mixing, late attention, hierarchical token reduction, and
deployment-time branch folding, but it does not copy Apple source code,
weights, image operators, or outputs.

## Development evidence

On the fixed Kim2025 development split, balanced accuracy for 0.4/0.6/0.8/1.0/
1.2 s was 10.94/22.60/36.15/48.96/56.88%. Under the identical split and
window protocol, the SSVEPformer reference produced
14.58/30.73/41.25/48.96/52.29%.

This supports a narrower claim: v2.3 closes the gap at 1.0 s and exceeds the
reference at 1.2 s while using 841,293 rather than 1,247,240 parameters. It
does not yet establish a short-window advantage.

## Decisions rejected during development

- v2.0 replaced the segmented candidate path and regressed.
- v2.1 restored that path and improved the result.
- v2.2 added learned physical-frequency position encoding and regressed.
- Dropout 0.3 did not beat dropout 0.1 on the registered validation score.

These alternatives remain recorded and must not be selected after seeing
external-test outcomes.

## External validation gate

The architecture and hyperparameters above must remain fixed for Benchmark,
BETA, and Wearable evaluation. A paper claim requires:

1. participant-disjoint results over registered seeds and folds;
2. the same channel, window, and split contract for neural baselines;
3. a separate calibrated table for TRCA/TDCA-family methods;
4. paired participant-level uncertainty and effect sizes;
5. full versus no-local/no-attention/no-evidence ablation evidence;
6. latency, parameter count, memory, and folded-graph equivalence.

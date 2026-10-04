# Final model decision v1

Date: 2026-10-03

## Decision

HarmonicFold spectral v4.1 is frozen as the final model for the current paper. No
later development candidate replaces it.

The decision follows a finite search that tested model width, temporal and
dual local domains, receptive field, nested spectral subbands, two forms of
reliability gating, duration conditioning, class-grid alignment, candidate
topology mixing in two placements, short-window schedule weighting,
long-to-short distillation, attention-path isolation, temporal segment
resolution and explicit phase-dynamics features. None produced a stable
0.4-second improvement while preserving 0.8 and 1.2 seconds on held-out
participants and then satisfying the independent-data rule.

## What the final model supports

- One universal set of weights accepts the registered 0.4--1.5-second windows.
- On Benchmark, relative to the protocol-matched SSVEPformer, v4.1 is lower at
  0.4 seconds, not distinguishable at 0.6 seconds after correction, and
  significantly higher at 0.8--1.2 seconds.
- On BETA, it is significantly lower at 0.4 seconds, tied at 0.6 seconds, and
  significantly higher at 0.8--1.2 seconds than SSVEPformer.
- Against FB-SSVEPformer on BETA, it is lower at 0.4--0.8 seconds, tied at 1.0
  second, and higher at 1.2--1.5 seconds.
- Relative to the MTSNet reconstruction, it loses at 0.4 seconds and is near
  parity at 0.8--1.2 seconds while using far fewer parameters.
- The folded model has 435,043 parameters and matches the training graph to
  within at most 1.43e-6 absolute logit error in the deployment benchmark.
- The 113-parameter target-user adapter provides a separate calibrated track;
  it must not be mixed with the zero-target-data comparisons.

## What the final model does not support

- universal superiority at 0.4 seconds;
- superiority over every SSVEPformer variant at every window;
- a universal fastest-inference claim;
- a claim that parameter count alone predicts wall-clock latency;
- a claim that the HarmonicFold visual backbone was copied unchanged into EEG;
- a claim that failed development candidates are part of the final method.

## Paper positioning

The defensible paper is a compact universal-window SSVEP decoder with a
local-to-global train/deploy topology, candidate-aligned
harmonic attention, measured mechanism evidence, low-parameter optional user
adaptation, and an accuracy/size/latency/memory Pareto. It is not positioned as
an all-window accuracy champion.

This is sufficient for a credible lightweight BCI or wearable neural-
engineering methods paper. The short-window limitation should be reported
directly; hiding it would weaken rather than strengthen the submission.

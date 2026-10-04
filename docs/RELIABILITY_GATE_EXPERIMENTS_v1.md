# Reliability-gate experiments v1

## Scope

These experiments test training-time learned, sample-adaptive control of the
global/harmonic attention residual. They do not use window-specific rules and
are not claimed as a standalone novelty.

## v4.5 implementation correction

The first v4.5 development run was invalid for its intended question: the
base candidate tensor had already been overwritten by the attention-refined
tensor before the two logits were compared. In evaluation mode the gate was
therefore interpolating identical paths. Revision `harmonic_fold_v4_5_1`
preserves the original candidate path and has a regression test proving that
low and high reliability select distinct base and attention paths.

On Benchmark grouped fold 0, seed 20260929, corrected v4.5 changed held-out
balanced accuracy relative to v4.1 by -0.60, -1.96, -1.49, +1.01 and +1.19
percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds. It fails the short-
window gate and is rejected.

## v4.6 isolated candidate-evidence gate

v4.4 confounded nested spectral subbands with a candidate-wise reliability
gate. v4.6 removes the subband intervention and adds only the candidate gate
to the frozen v4.1 backbone. Per candidate, the gate receives temporal
features, spectral features and their absolute disagreement, then scales the
attention residual. This is the direct test of whether same-trial evidence
agreement can suppress unreliable global/harmonic refinement without a hard-
coded window threshold.

On the same Benchmark development split, v4.6 changed held-out balanced
accuracy relative to v4.1 by +0.48, +0.89, -0.54, +1.01 and -0.77 percentage
points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds. It also reduced the mean
validation-selection score (0.68875 versus 0.69083) while adding 14,497
parameters. It therefore fails the no-material-long-window-regression gate and
is rejected without formal expansion.

Together, corrected v4.5 and isolated v4.6 show that learned reliability
control changes the short/long-window trade-off but does not dominate the
frozen v4.1 model. Further gate tuning is stopped to avoid development-set
search. The negative evidence does not alter the paper's v4.1 mainline.

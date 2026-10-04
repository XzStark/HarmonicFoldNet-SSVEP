# Width-64 accuracy/deployment Pareto protocol v1

Status: frozen before inspecting the development result.

## Question

The frozen v4.1 model uses width 48 and 435,139 parameters.  Stronger
SSVEPformer-family baselines use approximately 1.25--3.74 million parameters
and lead at the shortest windows.  Before adding another mechanism, test the
simpler explanation that v4.1 is capacity-limited.

## Candidate

Increase only the shared width from 48 to 64.  Keep the HarmonicFold local-first
topology, temporal and spectral candidate evidence, token reduction, harmonic
cross-attention, candidate attention, loss, optimizer, multi-window schedule
and evaluation protocol unchanged.  Only one larger width is tested; no width
sweep is allowed after viewing the result.

## Frozen development gate

Use the previously untouched Benchmark grouped fold 2, seed 20260929.  Compare
against frozen v4.1 on the identical split.  Advance only if 0.4-second held-
out balanced accuracy improves, 0.8 and 1.2 seconds do not materially regress,
mean registered-window validation selection does not fall, and total parameters
remain below one million.  A passing candidate must be frozen under a new
revision and evaluated on independent BETA with three seeds and five folds.

This experiment tests an accuracy/size Pareto point, not a new architectural
contribution.  Failure ends width scaling rather than triggering a post-hoc
width search.

## Result

Width 64 changed held-out balanced accuracy by -3.21, -2.26, +0.54, +2.98
and +3.27 percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds.  Mean
validation selection was essentially tied (+0.13 percentage points), while
parameters rose from 435,139 to 694,003.  It fails the registered short-window
gate and is rejected.

The direction of the changes indicates a representation/window trade-off, not
a simple lack of capacity.  Width scaling is therefore stopped as planned;
width 48 remains the mainline.

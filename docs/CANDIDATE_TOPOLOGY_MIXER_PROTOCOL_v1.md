# Candidate-frequency topology mixer protocol v1

Status: frozen before inspecting the development result.

## Motivation

At 0.4 seconds, neighboring SSVEP classes separated by only 0.2 Hz are not
cleanly resolved by a finite Fourier window.  The v4.1 decoder extracts one
token per stimulus candidate, but its candidate self-attention is global: it
has no explicit local operation that compares a candidate with its immediate
frequency neighbors before global fusion.

## Candidate

Keep the v4.1 temporal front end, temporal and spectral candidate extractors,
token reduction, harmonic cross-attention, candidate attention, loss and
training schedule unchanged.  Add one reparameterizable depthwise local block
over candidate tokens immediately before late attention.  Candidate tokens are
sorted by physical stimulus frequency (and phase only to break equal-frequency
ties), locally mixed with a three-point kernel, and restored to dataset class
order before scoring.

This is an EEG-specific continuation of the HarmonicFold local-first principle:
local candidate contrast precedes global attention, while the training-time
multi-branch mixer folds to one depthwise convolution for deployment.  It uses
no dataset names, window labels or hand-set frequency thresholds.

## Frozen development gate

Use Benchmark grouped fold 0 and seed 20260929, identical to the v4.1
development comparison.  Advance only if held-out 0.4-second balanced accuracy
improves, 0.8- and 1.2-second performance do not materially regress, the mean
registered-window validation score does not fall materially, the folded graph
matches the training graph, and the model stays below 500,000 parameters.

If it advances, freeze the architecture as v4.8 and run three seeds by five
grouped folds on independent BETA data.  If it fails, record it as negative
evidence and retain v4.1 as the paper mainline.

## Result

The candidate changed held-out balanced accuracy by +1.55, 0.00, -2.20,
+0.95 and +0.24 percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds.
The mean validation-selection score was effectively tied (-0.02 percentage
points), and parameters increased from 435,139 to 473,444.  The 0.8-second
regression violates the frozen gate, so this exact pre-cross placement is
rejected and is not promoted to v4.8.

Error analysis showed that the intervention changed both near-frequency and
distant-frequency decisions rather than selectively resolving adjacent
candidates.  This motivates a separately registered placement test; it does
not retroactively change the v1 result.

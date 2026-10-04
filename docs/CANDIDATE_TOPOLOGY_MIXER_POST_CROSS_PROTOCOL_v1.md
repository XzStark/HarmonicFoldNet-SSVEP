# Post-cross candidate topology mixer protocol v1

Status: frozen before inspecting the development result.

## Rationale

The first candidate-topology experiment mixed candidate embeddings before
harmonic cross-attention.  Although it improved 0.4-second accuracy, it also
changed distant-frequency decisions and reduced 0.8-second accuracy.  Mixing
the query before candidate-specific harmonic evidence is retrieved can blur
the semantic alignment between each query row and its harmonic-bias row.

This candidate preserves the same frequency-sorted, reparameterizable local
block but moves it after harmonic cross-attention and before global candidate
self-attention.  Thus each candidate first retrieves its own spectral/harmonic
evidence, then contrasts that enriched representation with adjacent physical
frequencies, and only then participates in global fusion.  All other v4.1
architecture and training settings remain unchanged.

## Frozen development gate

Use the previously untouched Benchmark grouped fold 1, seed 20260929.  Compare
against the already frozen v4.1 result on the identical split.  Advance only
if 0.4-second held-out balanced accuracy improves, neither 0.8 nor 1.2 seconds
regresses materially, mean registered-window validation selection does not
fall materially, folded and training graphs agree, and parameters remain below
500,000.  A passing candidate must then be frozen and tested on independent
BETA data with three seeds and five grouped folds before any primary claim.

## Result

On the frozen fold, held-out balanced accuracy changed by +1.67, -0.48,
-5.00, -4.88 and -4.41 percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2
seconds.  Mean validation selection also decreased by 0.42 percentage points.
The candidate therefore fails decisively and is not promoted.

Together with the pre-cross experiment, this shows that candidate-axis local
mixing can move the short-window operating point but does not preserve the
multi-window Pareto frontier.  Further topology-placement tuning is stopped to
avoid repeated development-set search; v4.1 remains the frozen mainline.

# Short-window front-end diagnosis protocol v1

Status: completed; candidate rejected and architecture search stopped.

## Question

The universal v4.1 model is competitive or better at medium windows but loses
to strong references at 0.4 seconds. Previous controlled experiments rule out
simple model width, duration conditioning, candidate-neighbour mixing,
attention removal and long-to-short distillation as stable solutions. The next
question is whether the temporal candidate front end discards short-window
evidence through its fixed four-way segmentation.

## Stage A: representation diagnosis

Keep architecture, optimizer, losses, participant split, train windows and
model selection identical to v4.1. On the frozen Benchmark development fold
and seed 20260929, train only the following temporal-demodulation resolutions:

- one whole-window segment;
- two equal segments;
- the frozen four-segment v4.1 reference;
- eight equal segments.

No result from this stage may be reported as final generalization evidence.
The comparison diagnoses whether short-window performance is limited primarily
by global signal-to-noise aggregation, by fine phase evolution, or whether
segment resolution is not the bottleneck.

### Stage A result

| Segments | Parameters | Selection mean | 0.4 s | 0.8 s | 1.2 s |
|---:|---:|---:|---:|---:|---:|
| 1 | 379,267 | 65.71% | 29.58% | 65.95% | 78.87% |
| 2 | 397,891 | 67.63% | 28.10% | 67.20% | 80.48% |
| 4, frozen v4.1 | 435,139 | 69.08% | 29.94% | 67.32% | 79.29% |
| 8 | 509,635 | 69.50% | 27.02% | 66.85% | 78.51% |

The apparent validation improvement from eight segments did not transfer to
held-out participants. One, two and eight segments all failed to improve the
registered 0.4-second endpoint. Segment count itself is therefore not promoted
as the solution.

## Stage B: one permitted architecture candidate

Stage A indicates that moderate temporal subdivision is useful but that finer
subdivision alone loses held-out short-window accuracy. The one permitted
candidate therefore retains the frozen four segments and adds five compact,
candidate-aligned phase-trajectory statistics per channel and harmonic:
mean adjacent-segment phase rotation (cosine and sine), first-to-last phase
rotation (cosine and sine), and log-amplitude instability. These quantities
are computed from the existing complex demodulation coefficients; they add no
filter-bank network, class-specific parameters, dataset thresholds or second
encoder. The projection remains shared across candidates.

The candidate alters only the temporal candidate representation. The spectral
path, foldable local front end, token reduction, harmonic-biased late
cross-attention, candidate attention, score head, loss and training schedule
remain frozen. The motivation is specific: raw real/imaginary segment values
force the MLP to learn bilinear phase differences from limited cross-subject
data, while explicit unit-complex products expose phase continuity without
changing the HarmonicFold local-to-global topology.

The candidate advances only if it improves the held-out 0.4-second endpoint,
does not materially regress 0.8 or 1.2 seconds, and preserves a compact folded
deployment graph. If it passes development, it receives an independent BETA
three-seed/five-fold evaluation. If it fails either gate, v4.1 remains final and
architecture search stops for this paper.

### Stage B result

On the frozen Benchmark development fold and seed, the phase-dynamics
candidate changed held-out balanced accuracy by -0.66, -1.43, -1.55, +2.26
and -0.66 percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds. Validation
selection was nearly tied (69.23% versus 69.08%), but the registered
0.4-second endpoint and both preservation windows failed. Parameters increased
from 435,139 to 466,179.

The candidate is rejected. It is not promoted to a named architecture, is not
run on BETA, and does not replace v4.1.

## Error-structure check and stopping decision

Across the complete BETA three-seed/five-fold predictions at 0.4 seconds,
HarmonicFold v4.1 errors were more frequency-local than random errors, but not
dominated by immediate neighbours: 11.57% were within one frequency rank,
17.17% within two and 37.34% within four; the median error was eight ranks
away. FB-SSVEPformer showed a similar pattern (12.50%, 18.91%, 39.03%; median
seven ranks). A local hard-negative objective would therefore target only a
minority of the remaining mistakes and is not justified as another development
candidate.

Together with the earlier negative width, reliability-gate, duration,
candidate-topology, schedule and distillation experiments, this closes the
registered architecture-search cycle. The frozen final paper model remains
HarmonicFold spectral v4.1. Further progress should use new data, a genuinely new
pretraining regime or a separately preregistered study rather than more
selection against the current development folds.

## Interpretation boundary

This protocol cannot prove that no future model could improve the task. It
defines a finite, falsifiable stopping point for the current HarmonicFold research
line and avoids selecting repeatedly against the same development subjects.

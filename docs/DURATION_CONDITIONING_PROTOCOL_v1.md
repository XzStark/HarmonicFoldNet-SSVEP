# Duration-conditioning protocol v1

Status: frozen before inspecting the development result.

## Motivation

The shared multi-window model standardizes every crop and divides Fourier
coefficients by the observed sample count. Its candidate and spectral token
shapes are also fixed by zero padding. Consequently, the decoder receives no
explicit indication of whether evidence came from 0.4 or 1.2 seconds, even
though the reliability of phase and harmonic estimates depends strongly on
the number of observed cycles.

## Candidate

Keep the v4.1 local mixer, candidate extractors, token reduction, attention and
training protocol unchanged. Add a small continuous conditioning vector to
each candidate token before late attention. It is computed from observed
duration and candidate-frequency cycles using bounded duration, log-duration,
bounded cycles and log-cycles features. The projection is zero-initialized, so
training starts exactly from the unconditioned topology.

No dataset identity, discrete window label or rule such as "disable attention
at 0.4 seconds" is supplied. The mapping accepts any positive sample count.
The candidate advances from Benchmark grouped fold 0, seed 20260929, only if
0.4-second held-out accuracy improves without material regression at 0.8 or
1.2 seconds and the mean registered-window selection score does not fall.

## Development result and freeze

Held-out balanced accuracy changed by +0.95, +0.36, -0.06, +1.85 and +0.71
percentage points at 0.4, 0.6, 0.8, 1.0 and 1.2 seconds. Parameters increased
from 435,139 to 444,931. The validation-selection mean was 0.17 percentage
points lower than v4.1, so the literal development gate was not fully met;
this uncertainty is recorded rather than hidden.

The architecture is now frozen as v4.7 without further tuning. A complete
three-seed, five-fold BETA evaluation will be treated as independent external
evidence. v4.7 replaces v4.1 only if BETA shows a repeatable 0.4-second gain,
no material 0.8/1.2-second regression, and an acceptable deployment-latency
change. Otherwise it remains a rejected development candidate.

## Independent BETA result

Across three seeds and five grouped folds, v4.7 changed held-out balanced
accuracy relative to v4.1 by -0.45, -0.46, +0.08, approximately 0.00, +0.56
and +1.37 percentage points at 0.4, 0.6, 0.8, 1.0, 1.2 and 1.5 seconds.
Holm-corrected paired testing found no short-window improvement; only the
1.5-second gain was significant.  The candidate therefore fails its stated
primary gate and is rejected.  Duration conditioning is not part of the final
mainline.

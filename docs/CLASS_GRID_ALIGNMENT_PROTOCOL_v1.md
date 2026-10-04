# Class-grid alignment protocol v1

Status: frozen before inspecting the development result.

## Motivation

Benchmark and BETA use 40 targets separated by 0.2 Hz. The frozen v4.1 front
end uses a 0.25-Hz spectrum and reduces spectral tokens at stride two, so the
late-attention grid is coarser than the target spacing. Zero padding does not
create new observations, but a grid coarser than the known candidate geometry
adds interpolation error and makes adjacent candidate neighborhoods nearly
identical.

## Candidate

Keep every v4.1 trainable module and training setting unchanged. Derive the
front-end resolution from the candidate set using a generic rule:

- spectrum resolution is at most half the minimum spacing between distinct
  candidate frequencies;
- harmonic-bias width is at most that minimum spacing.

Thus the dense 0.2-Hz Benchmark/BETA grids use 0.1-Hz spectrum bins and 0.2-Hz
post-downsampling tokens, while a 0.5-Hz-spaced set retains the existing
0.25-Hz spectrum and 0.5-Hz token grid. No dataset name or window threshold is
encoded in the model.

The candidate advances only if Benchmark grouped fold 0, seed 20260929,
improves 0.4 seconds without material 0.8/1.2-second regression. Because the
token count rises on dense candidate sets, any accepted accuracy result must
subsequently pass the same folded CPU/GPU latency protocol.

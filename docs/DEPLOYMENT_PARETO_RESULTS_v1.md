# Deployment Pareto results v1

Status: completed protocol-matched batch-1 benchmark on 2026-10-03.

## Measurement protocol

- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU and the host CPU.
- Batch size: 1; float32; CPU timings use one inference thread.
- Warm-up: 100 iterations; timed runs: 1,000 iterations.
- Windows: 0.4, 0.8 and 1.2 seconds.
- Reported latency includes each model's registered spectral preprocessing.
- HarmonicFold results use the folded deployment graph. SSVEPformer and
  FB-SSVEPformer are protocol-adapted reference implementations. MTSNet is a
  local reconstruction from the pinned upstream source and is not described
  as an exact reproduction of its published latency.

## Parameters and latency

| Model | Window | Parameters | CPU P50 / P95 (ms) | CUDA P50 / P95 (ms) | CUDA peak (MiB) |
|---|---:|---:|---:|---:|---:|
| HarmonicFold-SSVEP v4.1, folded | 0.4 s | 435,043 | 2.991 / 4.008 | 3.465 / 4.506 | 12.09 |
| HarmonicFold-SSVEP v4.1, folded | 0.8 s | 435,043 | 2.899 / 4.068 | 3.717 / 5.357 | 13.26 |
| HarmonicFold-SSVEP v4.1, folded | 1.2 s | 435,043 | 3.040 / 4.390 | 3.428 / 5.522 | 14.42 |
| SSVEPformer | 0.4 s | 1,247,240 | 0.874 / 1.173 | 2.611 / 3.258 | 14.03 |
| SSVEPformer | 0.8 s | 1,247,240 | 0.934 / 1.178 | 2.617 / 3.890 | 14.03 |
| SSVEPformer | 1.2 s | 1,247,240 | 0.826 / 1.149 | 2.528 / 3.656 | 14.04 |
| FB-SSVEPformer | 0.4 s | 3,741,724 | 2.756 / 3.593 | 7.601 / 9.602 | 23.57 |
| FB-SSVEPformer | 0.8 s | 3,741,724 | 2.750 / 4.134 | 7.743 / 11.764 | 23.58 |
| FB-SSVEPformer | 1.2 s | 3,741,724 | 2.673 / 4.270 | 7.753 / 11.734 | 23.58 |
| MTSNet reconstruction | 0.4 s | 8,718,800 | 5.512 / 7.159 | 3.179 / 4.969 | 43.26 |
| MTSNet reconstruction | 0.8 s | 10,484,200 | 6.889 / 8.309 | 2.888 / 4.064 | 49.63 |
| MTSNet reconstruction | 1.2 s | 12,569,600 | 7.588 / 9.409 | 2.801 / 3.649 | 57.60 |

The folded HarmonicFold graph matches the training graph within
`7.15e-7`--`1.43e-6` maximum absolute logit error. Folding reduces CUDA P50
relative to the training graph at every measured window, but does not make the
model faster than the much simpler SSVEPformer reference. Parameter count alone
is therefore not used as a latency claim.

## Supported interpretation

HarmonicFold-SSVEP v4.1 is a compact accuracy-resource Pareto point: it uses about
35% of SSVEPformer's parameters, 12% of the 0.8-second MTSNet reconstruction's
parameters, and much less CUDA memory than the filter-bank and MTSNet
references. It is faster than FB-SSVEPformer and MTSNet on one CPU thread, but
not faster than ordinary SSVEPformer. On this laptop GPU, small-kernel launch
overhead also prevents parameter count from translating directly to latency.

The defensible deployment contribution is consequently a measured
accuracy/size/latency/memory Pareto, not a universal fastest-model claim.

## Evidence files

- `runs/paper_v39/deployment/harmonic_fold_v4_1_{0.4,0.8,1.2}s.json`
- `runs/paper_v39/deployment/ssvepformer_{0.4,0.8,1.2}s.json`
- `runs/paper_v39/deployment/fb_ssvepformer_{0.4,0.8,1.2}s.json`
- `runs/paper_v39/deployment/mtsnet_{0.4,0.8,1.2}s.json`

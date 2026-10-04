# Reproducible result snapshot

This file records the first complete subject-disjoint experiment. It is an
engineering and research checkpoint, not a final paper result.

## Dataset and split

- Dataset: Kim2025BetaRange / NEMAR `nm000127` v1.0.2, CC BY 4.0.
- 40 participants, six sessions per participant, 40 classes per session.
- Input: PO7, PO3, POz, PO4, PO8, O1, Oz, and O2.
- Window: the complete five-second stimulation interval, resampled to 250 Hz.
- Train: participants 1-32, validation: 33-36, held-out test: 37-40.
- No participant occurs in more than one split.

## Accuracy

| Method | Validation Top-1 | Test Top-1 | Test Top-5 |
| --- | ---: | ---: | ---: |
| Fixed two-harmonic power | 52.40% | 79.06% | not measured |
| Learned candidate-frequency evidence | 66.25% | 88.23% | 95.73% |
| Full time/frequency fusion model | 71.46% | 88.65% | 95.73% |

The held-out test set contains 960 trials. Per-participant fusion accuracies are
80.83%, 99.58%, 85.00%, and 89.17%. The spread is material and must be reported;
the aggregate alone is not sufficient evidence of universal performance.

## Size and latency

- Train graph: 890,570 trainable parameters.
- Reparameterized deploy graph: 888,266 trainable parameters.
- NVIDIA RTX 5060 Laptop GPU, batch 1: 8.01 ms -> 5.20 ms.
- AMD Ryzen 9 8945HX, one CPU thread, batch 1: 8.37 ms -> 6.01 ms.
- Reparameterized and training-graph outputs passed numerical equivalence tests.

Latency is model compute time for an already available five-second EEG window;
it does not include signal acquisition time. Results are machine-specific and
must not be presented as phone or glasses latency.

## Current interpretation

The result supports continued research on physics-guided evidence fusion and
local-to-global structural reparameterization for efficient SSVEP decoding.
It does not yet establish a paper-level state of the art. Repeated participant
folds, FBCCA/TRCA and deep-learning baselines, shorter-window experiments,
statistical tests, and device-side benchmarks are still required.

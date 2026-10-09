# Reproducible paper result snapshot

This file summarizes the frozen paper evidence. It is an offline research record,
not a clinical, online-BCI, or target-device result. Exact participant-level data,
split assignments, uncertainty estimates, adjusted tests, and hashes are stored in
`paper/source_data/`.

## Evidence scope

- Five public datasets and 306 unique participants
- Participant-disjoint five-fold evaluation
- Three fitted seeds for the headline neural comparisons
- One HarmonicFoldNet checkpoint per dataset, fold, and seed covers the registered
  observation windows
- Benchmark and BETA are selection-aware; Dong2023 was independently reserved
  until the architecture and six-window analysis contract were frozen
- Wearable supplies a frozen dry/wet electrode-condition analysis

## Main selection-aware results

| Dataset | Window | HarmonicFoldNet | Protocol-adapted SSVEPformer | Difference |
| --- | ---: | ---: | ---: | ---: |
| Benchmark | 0.8 s | 69.53% | 65.42% | +4.11 points |
| Benchmark | 1.2 s | 78.30% | 73.88% | +4.42 points |
| BETA | 0.8 s | 59.77% | 56.95% | +2.82 points |
| BETA | 1.2 s | 68.70% | 63.08% | +5.62 points |
| BETA | 1.5 s | 73.23% | 65.20% | +8.03 points |

The strongest references remained preferable at 0.4 s. Against the MTSNet
protocol reconstruction, the 1.2 s differences on Benchmark and BETA were not
significant after multiplicity correction.

## Independently reserved external-dataset result

| Window | HarmonicFoldNet | SSVEPformer | MTSNet | CCA | FBCCA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 s | 15.99% | 17.93% | 17.33% | 6.42% | 4.94% |
| 0.6 s | 23.81% | 24.64% | - | 11.41% | 9.11% |
| 0.8 s | 32.98% | 30.84% | 32.31% | 18.28% | 15.41% |
| 1.0 s | 38.68% | 33.38% | - | 25.08% | 22.09% |
| 1.2 s | 43.62% | 34.58% | 41.74% | 31.89% | 29.74% |
| 1.5 s | 48.29% | 36.45% | - | 40.59% | 40.05% |

Relative to SSVEPformer, HarmonicFoldNet was 1.95 points lower at 0.4 s,
showed no detected difference at 0.6 or 0.8 s, and was 5.29, 9.04, and 11.84
points higher at 1.0, 1.2, and 1.5 s after Holm correction. Its MTSNet
differences at 0.4, 0.8, and 1.2 s were -1.35, +0.67, and +1.88 points; none
survived Holm correction. Each neural model was refitted inside each outer fold,
so this is external-dataset validation rather than zero-shot weight transfer.

## Mechanism and deployment evidence

- Spectral-token counts were 157, 79, and 40 for strides 1, 2, and 4.
- Stride 2 stayed within 0.20 points of no reduction at 0.4, 0.8, and 1.2 s,
  while reducing counted MACs by 11.4--12.6% and median single-thread CPU P50
  latency by 4.2--7.6%.
- The temporal candidate path contributed 1.30--2.64 points across all six BETA
  windows after Holm correction.
- The spectral-neighbourhood path had a weaker independent contribution; only
  the 1.5 s contrast was detected after correction.
- The folded graph contains 435,043 parameters versus 435,139 in the training
  graph and preserved every label across 49,000 held-out paired predictions.
- Five fresh-process timing sessions supported repeatable folding speedups only
  for CPU at 0.8 s and GPU at 0.8 and 1.2 s. Numerical equivalence is the
  unconditional result; runtime improvement is backend and window dependent.

## Interpretation boundary

The evidence supports a compact multi-window accuracy-resource trade-off from
approximately 0.8 s onward. It does not establish universal superiority,
cross-dataset zero-shot decoding, online communication rate, clinical utility,
or performance on glasses-mounted or peri-auricular electrodes.

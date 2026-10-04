# FB-SSVEPformer protocol-adapted baseline v1

Status: frozen before BETA results are inspected.

## Purpose and scope

FB-SSVEPformer is a required strong modern baseline, not part of the proposed
method.  The local implementation follows the public SSVEPformer topology and
the paper's three independently parameterized subnetwork plus learned score-
fusion design.  To satisfy the common input contract, all methods receive the
same eight channels, 250 Hz samples, causal evaluation windows, subject splits
and registered 6--45 Hz band.

The three deterministic spectral masks use lower cutoffs of 6, 14 and 22 Hz
within that common band.  This is explicitly a protocol adaptation, not a
bit-identical reproduction of the paper's 80 Hz IIR preprocessing.  No target-
participant trials are used for fitting or checkpoint selection.

## Training and evaluation

- datasets: Benchmark and BETA;
- three fixed random seeds: 20260929, 20260930 and 20260931;
- five fixed subject-disjoint grouped folds per seed;
- identical 0.4/0.6/0.8/1.0/1.2-second training schedule and validation-
  selection windows as the proposed model;
- the three subnetworks receive direct auxiliary class supervision while the
  learned fusion head is optimized on the same batches;
- primary statistics are participant-level out-of-fold balanced accuracy,
  participant bootstrap confidence intervals, paired tests, effect size and
  Holm correction across windows.

Benchmark was completed before this document and remains labelled as prior
baseline work.  The BETA 15-run matrix is the remaining confirmatory baseline
gap; its settings above are frozen before aggregate BETA results are viewed.

## BETA result

All 15 runs completed and cover the same 70 held-out participants for both
conditions.  Participant-level three-seed balanced accuracy was:

| Window | HarmonicFold v4.1 | FB-SSVEPformer | v4.1 minus FB |
|---:|---:|---:|---:|
| 0.4 s | 27.37% | 34.06% | -6.69 pp |
| 0.6 s | 43.47% | 48.57% | -5.10 pp |
| 0.8 s | 55.60% | 57.89% | -2.29 pp |
| 1.0 s | 63.56% | 63.78% | -0.22 pp |
| 1.2 s | 68.70% | 67.30% | +1.40 pp |
| 1.5 s | 73.12% | 70.78% | +2.34 pp |

After Holm correction, FB-SSVEPformer leads significantly at 0.4--0.8 seconds,
the 1.0-second difference is not significant, and v4.1 leads significantly at
1.2 and 1.5 seconds.  The correct claim is therefore a window-dependent
accuracy/efficiency Pareto, not universal superiority.  v4.1 uses 435,139
parameters versus 3,741,724 for the protocol-adapted FB-SSVEPformer.

Machine-readable comparison:
`runs/paper_v36/comparisons/beta_harmonic_fold_vs_fb_ssvepformer_3seed.json`.

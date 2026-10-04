# Wearable dry/wet electrode transfer results v1

Status: frozen confirmatory result for the fixed `harmonic_fold_v4_1`
model. This document reports a public-dataset proxy for electrode-domain
shift; it is not evidence from the project's intended glasses electrodes.

## Protocol

- Dataset: Wearable SSVEP, 102 participants and 12 classes.
- Channels: the eight posterior channels frozen in
  `configs/paper_multidataset.yaml`.
- Evaluation: five-fold subject-disjoint outer cross-validation. Every target
  participant is absent from both source-condition training and validation.
- Seeds: `20260929`, `20260930`, and `20260931`, with identical outer folds.
- Model and optimizer: frozen before this directional-transfer experiment.
- Statistical unit: participant. Each participant is first averaged across
  the three seeds; paired tests are then performed across 102 participants.
- Multiple testing: Holm correction across all registered transfer contrasts
  and windows.
- Accuracy below is balanced accuracy. Confidence intervals are 95% bootstrap
  intervals for the participant mean. `Seed SD` is the mean within-participant
  standard deviation across the three training seeds.

## Primary 0.4--1.2 second results

| Train -> test | Window (s) | Accuracy (%) | 95% CI (%) | Seed SD (pp) |
| --- | ---: | ---: | ---: | ---: |
| dry -> dry | 0.4 | 27.63 | 24.92--30.53 | 2.12 |
| dry -> dry | 0.6 | 36.40 | 32.92--40.01 | 2.36 |
| dry -> dry | 0.8 | 44.73 | 40.69--48.78 | 2.54 |
| dry -> dry | 1.0 | 51.16 | 46.85--55.51 | 2.64 |
| dry -> dry | 1.2 | 55.05 | 50.53--59.54 | 2.53 |
| dry -> wet | 0.4 | 35.65 | 32.41--38.97 | 2.04 |
| dry -> wet | 0.6 | 46.35 | 42.24--50.45 | 2.12 |
| dry -> wet | 0.8 | 54.58 | 50.15--58.97 | 2.33 |
| dry -> wet | 1.0 | 59.85 | 55.19--64.43 | 2.12 |
| dry -> wet | 1.2 | 64.25 | 59.66--68.68 | 2.16 |
| wet -> dry | 0.4 | 24.00 | 21.10--27.11 | 2.19 |
| wet -> dry | 0.6 | 30.48 | 26.77--34.38 | 2.52 |
| wet -> dry | 0.8 | 35.30 | 30.94--39.90 | 2.63 |
| wet -> dry | 1.0 | 38.97 | 34.22--43.93 | 2.55 |
| wet -> dry | 1.2 | 42.29 | 37.25--47.50 | 2.84 |
| wet -> wet | 0.4 | 45.19 | 41.48--48.89 | 2.56 |
| wet -> wet | 0.6 | 56.45 | 52.23--60.68 | 2.42 |
| wet -> wet | 0.8 | 64.30 | 59.91--68.63 | 2.35 |
| wet -> wet | 1.0 | 69.06 | 64.66--73.35 | 2.29 |
| wet -> wet | 1.2 | 72.64 | 68.34--76.79 | 2.34 |

## Paired transfer effects

Differences are the left condition minus the right condition, in percentage
points. The effect size is the paired rank-biserial correlation.

| Contrast | Window (s) | Difference (pp) | Holm-adjusted p | Rank-biserial |
| --- | ---: | ---: | ---: | ---: |
| dry -> wet minus wet -> wet | 0.4 | -9.55 | 1.236e-13 | -0.890 |
| dry -> wet minus wet -> wet | 0.6 | -10.10 | 2.840e-13 | -0.878 |
| dry -> wet minus wet -> wet | 0.8 | -9.72 | 2.044e-12 | -0.851 |
| dry -> wet minus wet -> wet | 1.0 | -9.22 | 4.829e-12 | -0.845 |
| dry -> wet minus wet -> wet | 1.2 | -8.39 | 1.121e-11 | -0.827 |
| wet -> dry minus dry -> dry | 0.4 | -3.63 | 2.110e-3 | -0.352 |
| wet -> dry minus dry -> dry | 0.6 | -5.92 | 5.488e-4 | -0.417 |
| wet -> dry minus dry -> dry | 0.8 | -9.44 | 3.700e-6 | -0.566 |
| wet -> dry minus dry -> dry | 1.0 | -12.19 | 3.072e-7 | -0.627 |
| wet -> dry minus dry -> dry | 1.2 | -12.76 | 1.313e-6 | -0.591 |
| dry -> wet minus wet -> dry | 0.4 | +11.64 | 7.030e-9 | +0.703 |
| dry -> wet minus wet -> dry | 0.6 | +15.86 | 2.970e-10 | +0.758 |
| dry -> wet minus wet -> dry | 0.8 | +19.29 | 4.128e-11 | +0.794 |
| dry -> wet minus wet -> dry | 1.0 | +20.87 | 2.440e-11 | +0.804 |
| dry -> wet minus wet -> dry | 1.2 | +21.96 | 2.404e-11 | +0.806 |

## Interpretation boundary

The result establishes a repeatable, asymmetric electrode-domain shift under
subject-disjoint evaluation. Wet recordings are easier as a target domain,
and training on dry recordings transfers to wet recordings better than the
reverse direction. Neither direction matches its target-domain same-condition
control, so the current model must not be described as electrode-domain
invariant.

This is useful evidence for the wearable research question: robustness cannot
be inferred from a pooled dry/wet score, and intended peri-auricular or glasses
electrode placement will require its own paired acquisition and adaptation
study. It does not establish performance for SSVEP glasses hardware that has
not yet been measured.

## Machine-readable evidence

The authoritative aggregate is:

`runs/paper_v16/electrode_transfer/3seed_summary.json`

It includes the registered 1.5 s and 2.0 s secondary windows in addition to the
primary table above. The 60 underlying fold results are stored under the four
condition directories in `runs/paper_v16/electrode_transfer/`.

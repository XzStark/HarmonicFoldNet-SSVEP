# Author-side simulated review, round 1

This document records an internal pre-submission stress test. It is not independent
peer review and must not be presented as editorial acceptance or external validation.

## Editorial fit

**Recommendation before revision: major revision.** The manuscript is within the
scope of *Journal of Neural Engineering*: it studies an EEG decoder, subject-level
evaluation, optional calibration, electrode-condition shift, and deployment cost.
The credible paper claim is a compact, multi-window decoder with a useful medium-window
accuracy/resource trade-off. It is not universal state of the art and does not win at
0.4 s.

## Methodology and statistics

The main risks were protocol overstatement and non-independent observations. The
revised manuscript now treats the participant as the unit of inference, averages seed
repeats within participant, declares Holm families, and reports participant-bootstrap
intervals. Benchmark and BETA are explicitly selection-aware because later candidate
screening consulted them. Wearable is described as frozen dry/wet acquisition-domain
analysis rather than proof of glasses-mounted generalization.

The calibrated comparison is retrospective cyclic leave-one-block evaluation, not
prospective online calibration. The neural adapter and classical methods receive the
same number of target-labelled blocks, but only the adapter inherits source-participant
pretraining; the text now calls this equal target-label burden rather than equal total
supervision.

## Comparator fairness

CCA is retained only as an interpretable historical lower-bound control. The comparison
set also contains FBCCA, TRCA, ensemble TRCA, TDCA, SSVEPformer, FB-SSVEPformer, and a
pinned-source MTSNet reconstruction. All locally executed methods use the declared
eight-channel, 250 Hz signal contract where their architecture permits. MTSNet requires
one model per window, whereas HarmonicFoldNet uses one checkpoint across windows; its
comparison is therefore presented as an operational trade-off rather than an isolated
architectural effect.

## Mechanism claims

The harmonic-attention audit measures allocation imposed by a fixed bias. It is not a
faithful explanation of individual predictions or evidence of a newly discovered
physiological mechanism. Trial–head observations are used only to summarize attention
location; participant-level paired quantities are used for tests. The no-attention
ablation removes the combined late cross-attention and candidate-attention path and is
labelled accordingly. The independently trained no-bias ablation supports an accuracy
benefit only at 1.0 and 1.5 s.

## Deployment and reproducibility

The revised deployment claim is bounded: HarmonicFoldNet has fewer parameters and less
memory than the evaluated neural references and lower CPU latency than MTSNet, but it is
not the fastest model in every comparison. The expanded fold-equivalence audit covers
20 checkpoints, five windows, and 49,000 held-out paired predictions with zero label
disagreements. Device, precision, batch size, thread count, warm-up, and repetitions are
reported.

## Remaining author-supplied blockers

- Affiliation and corresponding-author email.
- Funding and competing-interest declarations.
- Permanent GitHub/Hugging Face addresses and a frozen release tag.
- Final arXiv identifier after the preprint is posted.

Subject to those administrative fields and a clean public-artifact audit, the manuscript
is ready for formatting and submission rather than another architecture search.

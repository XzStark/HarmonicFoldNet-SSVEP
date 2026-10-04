# HarmonicFoldNet publication positioning

Date: 2026-10-03

## Name decision

Retain **HarmonicFoldNet**. Exact-name searches found no meaningful collision
in the scholarly/model records screened on the date above. The name describes
two actual properties of the method: harmonic-conditioned evidence and
train-to-deploy structural folding. The first occurrence in the manuscript
must define "Fold" explicitly so that readers do not mistake it for a
cross-validation fold.

Recommended title:

> HarmonicFoldNet: A Compact Foldable Local-to-Global Decoder for
> Cross-Subject SSVEP Recognition

## Current scientific position

The work is stronger than a small architecture variant or student benchmark
demo. Its defensible contribution is the combined method-and-evidence package:

1. candidate-aligned local-to-global decoding with token reduction before late
   global attention;
2. harmonic-conditioned attention supported by an isolated ablation and
   attention-mass analysis;
3. one weight set spanning all registered observation windows;
4. numerically verified training-to-deployment folding;
5. complete grouped participant folds, three seeds, multiple public datasets,
   modern learned baselines and separately reported calibrated baselines;
6. an optional 113-parameter target-user adaptation path; and
7. measured accuracy, parameter, memory and latency trade-offs.

The evidence does not support an all-window state-of-the-art claim. The main
model is weaker at 0.4 seconds, does not introduce a new mathematical theory,
and has not yet been validated in an online study using the intended wearable
electrode hardware. The paper should therefore claim a compact,
mechanism-supported deployment Pareto rather than universal superiority.

## Venue fit

### Recommended first submission

**Biomedical Signal Processing and Control** is the best current fit. Its
scope explicitly includes EEG, BCI, deep learning, wearable systems,
personalization, interpretability, real-time operation and embedded
implementation. The present offline multi-dataset evidence is aligned with
that scope.

### Ambitious submissions

**Journal of Neural Engineering** is a credible stretch target. Its scope
directly covers BCI and neural signal processing, but it states that modest
classification gains on small public datasets are insufficient without modern
comparisons, additional datasets and insight into the mechanism. The project
already addresses these three points; real-device or online validation would
materially strengthen the submission.

**IEEE Transactions on Neural Systems and Rehabilitation Engineering** and
**IEEE Journal of Biomedical and Health Informatics** are also possible but
harder. They are most defensible after an online/wearable experiment or a
stronger practical calibration study. Recent cross-subject Transformer work in
JBHI makes the reviewer bar particularly high.

### Alternative and fallback

**Neurocomputing** is a reasonable architecture-oriented alternative if the
paper emphasizes the compact hybrid network and ablations. **IEEE EMBC 2027**
is a sound archival conference route if an earlier, shorter publication and
review feedback are preferred.

## Claim boundary after the refreshed search

Local convolution plus attention, harmonic features, harmonic attention,
time-frequency fusion, cross-subject decoding and EEG structural
reparameterization all have prior art. The paper must not claim any one of
those ingredients as first. The publishable novelty is the particular topology
and its unusually complete evaluation package, not a newly invented primitive.

This document is a literature-scoping and venue-positioning assessment. It is
not a systematic review, acceptance prediction, trademark clearance or patent
freedom-to-operate opinion.

## Sources

- SSVEPformer: https://doi.org/10.1016/j.neunet.2023.04.045
- DG-Conformer: https://pubmed.ncbi.nlm.nih.gov/39226201/
- MTSNet: https://pubmed.ncbi.nlm.nih.gov/40408213/
- Journal of Neural Engineering scope:
  https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
- Biomedical Signal Processing and Control scope:
  https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094
- Scientific workflow support: Kassis T, Agarwal V, He Y, Patel D, Brueckner
  AM. *Scientific Agent Skills: A Library of Procedural Knowledge for Research
  Agents*. arXiv:2609.00065v2 (2026).

---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - eeg
  - ssvep
  - brain-computer-interface
  - time-series
  - cross-subject
  - structural-reparameterization
datasets:
  - BETA SSVEP
---

# HarmonicFoldNet

HarmonicFoldNet is a compact multi-window PyTorch decoder for cross-subject SSVEP
recognition. The frozen paper architecture combines foldable local temporal mixing,
candidate-aligned temporal and complex spectral evidence, reduced spectral tokens,
and harmonic-biased late attention.

This model repository is a noncommercial research release. It is not a medical
device and does not decode unrestricted thoughts or imagined language.

## Evaluation scope

This snapshot contains exactly 15 BETA source-decoder checkpoints: three seeds
by five participant-disjoint folds. It does not contain Benchmark, component-
ablation, Wearable, comparator, or participant-adapter checkpoints. The broader
paper evaluates four public datasets, but that paper-level scope must not be
mistaken for the weight scope of this model repository.

- Four public datasets; 247 unique participants in total
- Participant-disjoint five-fold evaluation
- Three training seeds for final neural comparisons
- Registered windows from 0.4 s to 1.2/1.5 s for headline comparisons
- One checkpoint handles every registered window within a dataset/fold/seed
- Optional 113-parameter participant adapter evaluated retrospectively

At 1.2 s, participant-level balanced accuracy was 78.30% on Benchmark and 68.70%
on BETA. The model did not lead at 0.4 s. Full confidence intervals, paired tests,
strong-reference comparisons, and selection-history qualifications are in the
paper source-data package; isolated headline numbers should not be treated as a
universal ranking.

## Deployment graph

- Training graph: 435,139 parameters
- Folded graph: 435,043 parameters
- Expanded equivalence audit: 20 checkpoints, five windows, 49,000 held-out paired
  predictions, zero label disagreements
- Maximum absolute logit error after folding: 1.55e-5

Laptop timings are implementation-specific and exclude EEG acquisition time. The
1.2 s folded graph measured about 3.04 ms median on one CPU thread in the reported
environment.

## Loading a released checkpoint

The minimal implementation is included, so a downloaded model snapshot does
not depend on a parent source checkout:

```python
from load_model import load_harmonic_fold_checkpoint

model, metadata = load_harmonic_fold_checkpoint(
    "checkpoints/beta/seed-20260929/fold-0/model.pt",
    folded=True,
)
```

Checkpoint loading uses PyTorch's restricted `weights_only=True` path. The
manifest and `SHA256SUMS.txt` should be verified before loading files obtained
from an untrusted transport.

## Intended use

- Inspect or re-evaluate the released BETA fold checkpoints
- Study compact multi-window SSVEP decoding
- Inspect train-to-deploy structural folding
- Evaluate calibration and acquisition-domain shifts in noncommercial research

## Out-of-scope use

- Medical diagnosis or treatment
- Emergency or safety-critical control
- Covert monitoring or identity inference
- Claims of general thought, intention, or inner-speech decoding
- Claims about peri-auricular or glasses-mounted EEG without new validation

## Data

Raw EEG is not included. Obtain Benchmark, BETA, Wearable SSVEP, and Kim2025 from
their original records for paper-level reproduction. The released checkpoints
were trained on BETA. Dataset licenses remain independent of this model license;
see `LICENSE_PROVENANCE_MATRIX.md`.

## Code and reproducibility

Source, exact configurations, participant-level derived results, and tests are at:

https://github.com/XzStark/FastSSVEPFusionNet

The public repository name is historical; the paper model and release name are
HarmonicFoldNet. See `paper/REPRODUCIBILITY_CHECKLIST.md` before comparing results.

## License

Weights and this model card are licensed under CC BY-NC 4.0. Authored source code is
distributed separately under PolyForm Noncommercial 1.0.0. This is source-available
research software rather than an OSI-approved open-source release.

## Citation

Use the repository `CITATION.cff` until the arXiv identifier is available. Do not
invent a venue, DOI, or peer-review status.

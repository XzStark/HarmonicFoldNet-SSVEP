# Third-party notices

This repository contains authored research code and derived metrics. It does not
redistribute raw EEG or the unlicensed implementation of an external comparison
model. Dataset and third-party terms remain independent of the repository licenses.

## Public datasets

### Kim2025 beta-range SSVEP

- Record: NEMAR `nm000127`, version 1.0.2
- Dataset DOI: https://doi.org/10.82901/nemar.nm000127
- Source-data DOI: https://doi.org/10.6084/m9.figshare.28806815.v2
- Article: https://doi.org/10.1038/s41597-025-06032-2
- Recorded dataset license: CC BY 4.0

### Tsinghua Benchmark SSVEP

- Portal: https://bci.med.tsinghua.edu.cn/
- Article: https://doi.org/10.1109/TNSRE.2016.2627556

### BETA SSVEP

- Portal: https://bci.med.tsinghua.edu.cn/
- Record: https://doi.org/10.6084/m9.figshare.12264401
- NEMAR record: https://nemar.org/dataset/nm000129
- Article: https://doi.org/10.3389/fnins.2020.00627
- Recorded terms checked 2026-10-04: non-commercial research use

### Wearable SSVEP

- Record: https://doi.org/10.6084/m9.figshare.13560281.v4
- Article: https://doi.org/10.3390/s21041256

Users must obtain each dataset from its original host and comply with the terms
displayed there. No dataset license is replaced or extended by `LICENSE` or
`MODEL_LICENSE.md`.

## External MTSNet comparison

The manuscript includes measurements from a protocol reconstruction that loads the
MTSNet authors' model definition from:

- Repository: https://github.com/lanzhen19/MTSNet
- Pinned commit: `890b0a4f93c3affd50741a1f1d036fb74a30a062`

No redistribution license was found in the pinned upstream source during the local
audit. The upstream implementation is therefore excluded. `src/external_mtsnet.py`
contains only the independently authored loading and protocol adapter; users must
obtain the authors' source themselves and follow any terms supplied by its owners.

## Baseline implementations

CCA, FBCCA, TRCA, ensemble TRCA, TDCA, SSVEPformer, and FB-SSVEPformer are local
research implementations or protocol adaptations written for this evaluation. Their
scientific sources are cited in `paper/references.bib`. Repository code is not a
claim of authorship over the underlying published methods.

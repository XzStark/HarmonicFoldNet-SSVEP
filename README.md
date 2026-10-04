# HarmonicFoldNet

HarmonicFoldNet is a compact multi-window decoder for cross-subject steady-state
visual evoked potential (SSVEP) recognition. It combines foldable local temporal
mixing, candidate-aligned temporal and complex spectral evidence, reduced spectral
tokens, harmonic-biased late attention, and an optional 113-parameter participant
adapter.

This repository is the reproducibility source for the HarmonicFoldNet preprint. It
is a **public noncommercial research release**, not an OSI-approved open-source
package, a medical device, or evidence of unrestricted thought decoding.

## Evidence at a glance

The paper evaluates four public datasets containing 247 unique participants. All
neural headline results use participant-disjoint outer folds and three training
seeds. Benchmark and BETA are explicitly reported as selection-aware because later
candidate screening consulted them; the Wearable dry/wet analysis was frozen after
model retention.

At 1.2 s, HarmonicFoldNet reached 78.30% balanced accuracy on Benchmark and 68.70%
on BETA. It was not the best method at 0.4 s. On BETA it crossed the ordinary
spectral Transformer after 0.6 s and was 1.40 percentage points above the
filter-bank Transformer at 1.2 s. Against the MTSNet protocol reconstruction, the
1.2 s differences were not significant on either Benchmark or BETA.

The folded deployment graph has 435,043 executable parameters. On the measured
laptop it required about 3.04 ms median batch-one CPU time at 1.2 s with one thread.
An expanded audit over 20 checkpoints, five windows, and 49,000 held-out prediction
pairs produced zero label disagreements between training and folded graphs; maximum
absolute logit error was 1.55e-5.

These are offline research measurements, not target-device or online BCI latency.

## What is included

- `src/harmonic_fold.py`: HarmonicFoldNet implementation
- `src/paper_train.py`: grouped participant training and evaluation entry point
- `src/paper_baselines.py` and `src/calibrated_baselines.py`: analytic and calibrated controls
- `configs/paper_multidataset.yaml`: frozen dataset and model contract
- `paper/MANUSCRIPT_DRAFT_v1.md`: manuscript source
- `paper/SUPPLEMENTARY_INFORMATION.md`: supplementary methods and tables
- `paper/source_data/`: participant-level derived results and provenance
- `paper/figures/`: publication figures and figure-source tables
- `paper/REPRODUCIBILITY_CHECKLIST.md`: release and reporting checklist
- `scripts/`: evidence reconstruction, audit, figure, and release helpers
- `tests/`: model, statistics, evidence, and deploy-equivalence tests

Raw EEG is not redistributed. Obtain each dataset from its original record and
retain its license and citation.

## Environment

The frozen evidence environment used Python 3.11.9 and PyTorch 2.11.0 with CUDA
12.8. Exact core versions are in `paper/environment-lock.txt`. For a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare public datasets

After downloading Benchmark, BETA, or Wearable SSVEP from the official source,
convert it to the registered shard format:

```powershell
.\.venv\Scripts\python.exe -m src.public_ssvep_data beta `
  --root D:\datasets\BETA `
  --output-dir data\processed\BETA
```

Use `benchmark` or `wearable` in place of `beta` for the other datasets. Kim2025
preparation is documented by `python -m src.kim2025_data --help`.

## Reproduce one outer fold

```powershell
.\.venv\Scripts\python.exe -m src.paper_train `
  --config configs\paper_multidataset.yaml `
  --dataset beta `
  --architecture harmonic_fold_v4_1 `
  --mode full `
  --seed 20260929 `
  --fold-index 0 `
  --fold-count 5 `
  --device cuda `
  --run-dir runs\reproduce\beta\seed-20260929\fold-0 `
  --save-checkpoint
```

The manuscript matrix uses seeds 20260929, 20260930, and 20260931 over all five
folds. The evidence configuration records every preserved input artifact:

```powershell
.\.venv\Scripts\python.exe -m scripts.rebuild_submission_from_bundle
.\.venv\Scripts\python.exe -m scripts.build_supplement
.\.venv\Scripts\python.exe -m scripts.audit_manuscript
.\.venv\Scripts\python.exe -m pytest -q
```

## Comparison boundary

CCA is retained as an interpretable historical lower bound, not the principal
benchmark. The paper also evaluates FBCCA, TRCA, ensemble TRCA, TDCA,
SSVEPformer, FB-SSVEPformer, and a pinned-source MTSNet reconstruction. The MTSNet
repository did not provide a redistribution license when audited; its source is
therefore not included here. The adapter, commit identifier, and protocol are
included so users can obtain the upstream source themselves under its owners' terms.

## License

- Authored source code: PolyForm Noncommercial 1.0.0 (`LICENSE`)
- Model weights and authored documentation: CC BY-NC 4.0 (`MODEL_LICENSE.md`)
- Public EEG datasets: their original independent terms

Commercial use requires separate written permission. Because the code license
restricts commercial use, describe this as a **source-available research release**
rather than OSI open source.

## Safety and limitations

HarmonicFoldNet decodes known visual-stimulation classes from registered posterior
EEG montages. It has not been validated for diagnosis, treatment, covert monitoring,
imagined speech, arbitrary mental-state inference, or safety-critical control. The
Wearable result compares public dry and wet recordings; it does not establish
performance for a future peri-auricular or glasses-mounted montage.

## Citation

Use `CITATION.cff` for the software release. Replace the software citation with the
arXiv or journal citation after the preprint identifier is available.

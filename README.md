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

The paper evaluates five public datasets containing 306 unique participants. All
neural headline results use participant-disjoint outer folds and three training
seeds. Benchmark and BETA are explicitly reported as selection-aware because later
candidate screening consulted them. Dong2023 was reserved until the architecture,
preprocessing, endpoints, and multiplicity plan were frozen, and supplies an
independent external-dataset validation. The Wearable dry/wet analysis was frozen
after model retention.

At 1.2 s, HarmonicFoldNet reached 78.30% balanced accuracy on Benchmark and 68.70%
on BETA. It was not the best method at 0.4 s. On BETA it crossed the ordinary
spectral Transformer after 0.6 s and was 1.40 percentage points above the
filter-bank Transformer at 1.2 s. Against the MTSNet protocol reconstruction, the
1.2 s differences were not significant on either Benchmark or BETA.

On Dong2023, HarmonicFoldNet was 1.95 percentage points below SSVEPformer at
0.4 s, showed no detected difference at 0.6 or 0.8 s, and was 5.29--11.84 points
higher at 1.0--1.5 s after Holm correction. Its differences from MTSNet at 0.4,
0.8, and 1.2 s were -1.35, +0.67, and +1.88 points; none was significant after
Holm correction. Each neural model was fitted inside each Dong2023 outer fold, so
this is external-dataset validation rather than zero-shot weight transfer.

The folded deployment graph has 435,043 executable parameters. On the measured
laptop it required about 3.04 ms median batch-one CPU time at 1.2 s with one thread.
An expanded audit over 20 checkpoints, five windows, and 49,000 held-out prediction
pairs produced zero label disagreements between training and folded graphs; maximum
absolute logit error was 1.55e-5.

Post-freeze mechanism checks compare spectral-token strides 1, 2, and 4; isolate
the temporal and spectral candidate-evidence paths; and repeat train-to-deploy
timing in five fresh processes per backend-window pair. Stride 2 halves the token
count relative to no reduction while staying within 0.20 points at 0.4, 0.8, and
1.2 s. The temporal candidate path contributes across all tested windows, whereas
an independent spectral-path contribution is detected only at 1.5 s. Folding is
prediction-equivalent, but repeatable speedups are backend and window dependent.

These are offline research measurements, not target-device or online BCI latency.

## What is included

- `src/harmonic_fold.py`: HarmonicFoldNet implementation
- `src/paper_train.py`: grouped participant training and evaluation entry point
- `src/paper_baselines.py` and `src/calibrated_baselines.py`: analytic and calibrated controls
- `configs/paper_multidataset.yaml`: frozen dataset and model contract
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
preparation is documented by `python -m src.kim2025_data --help`. Dong2023
provenance, split hashes, and derived comparison records are included in the paper
source-data manifest; raw EEG is obtained from the original Zenodo record.

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

Use `CITATION.cff` for the software release. The versioned preprint family is
archived under Zenodo concept DOI `10.5281/zenodo.23170146`; cite the journal
article instead if and when it is published.

# MTSNet external-baseline protocol (draft v0)

MTSNet is a close 2025 calibration-free temporal-spectral Convformer baseline.
The authors released model definitions at
<https://github.com/lanzhen19/MTSNet>, but the repository currently has no
license and does not include its training or preprocessing pipeline. Its open
reproducibility issue also records unresolved channel-selection, spectral
normalization, seed and checkpoint-selection details.

Accordingly, this repository must not copy or redistribute the upstream model
source. `src.external_mtsnet.ExternalMTSNetAdapter` imports an explicitly
provided local checkout for research-only comparison and records its source
commit. Release artifacts must omit `.research_refs/MTSNet`.

## Frozen reconstruction assumptions

- Use the authors' unmodified `ViT_MTFNet` class at commit
  `890b0a4f93c3affd50741a1f1d036fb74a30a062`.
- Use the released example values: two local blocks, two fusion blocks,
  kernel length 31 and dropout 0.5.
- Use the same eight posterior channels, participant folds, causal filtering,
  crop onset and per-trial time-domain standardization as the main protocol.
- Train an independent fixed-input model for each evaluated window because the
  released classifier head depends on temporal length.
- Reconstruct the advertised 560-dimensional complex spectrum as a 560-point
  real FFT, discard DC, then concatenate real and imaginary values from the
  remaining 280 bins.
- Primary spectral normalization is `none`. A pre-registered
  `channel_zscore` sensitivity analysis may be reported separately; it must not
  be selected on target-test performance.
- Optimizer selection was restricted to a Kim2025 development comparison:
  AdamW learning rates `5e-4` and `1e-4`, crossed with `none` and
  `channel_zscore` spectral normalization at 0.8 s. The two surviving `1e-4`
  candidates were then checked on a second development fold with up to 80
  epochs and patience 12. No Benchmark or BETA result was used for selection.
- The frozen formal configuration is AdamW `1e-4`, weight decay `0.02`, label
  smoothing `0.05`, batch size 128, no spectral normalization, at most 80
  epochs, and early-stopping patience 12.

## Development selection record

| Candidate | Kim fold 0 validation | Kim fold 1 validation |
| --- | ---: | ---: |
| `none`, lr `1e-4` | 53.54% | 36.46% |
| `channel_zscore`, lr `1e-4` | 53.85% | 35.52% |

The two-fold mean was 45.00% without spectral z-scoring and 44.69% with it.
The `5e-4` candidates were eliminated on fold 0 (51.77% without z-scoring and
49.48% with it). These values are development diagnostics, not paper test
results.

Any result produced under these assumptions is labelled **protocol
reconstruction**, not exact reproduction of the published MTSNet number. It is
usable as a same-data engineering baseline only after the reconstruction is
validated on an untouched development subset and all assumptions are reported.

Primary sources:

- Paper DOI: <https://doi.org/10.1109/JBHI.2025.3573410>
- Author repository: <https://github.com/lanzhen19/MTSNet>
- Open reproducibility issue: <https://github.com/lanzhen19/MTSNet/issues/1>

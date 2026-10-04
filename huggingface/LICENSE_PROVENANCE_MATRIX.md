# Dataset, code and weight licence provenance

Checked: 2026-10-04. This is a release audit, not legal advice. Dataset terms
remain controlled by the authoritative host and must be rechecked before each
public release.

| Asset | Use in this work | Authoritative record or source | Recorded terms | Redistributed? | Release consequence |
|---|---|---|---|---|---|
| Kim2025 beta-range EEG | Architecture development | NEMAR `nm000127`; DOI `10.82901/nemar.nm000127` | CC BY 4.0 | No raw EEG or Kim-trained weights | Cite dataset; users obtain it independently |
| Tsinghua Benchmark EEG | Selection-aware evaluation | Tsinghua BCI Laboratory portal; Wang et al. 2017 | Portal terms govern access; no licence identifier asserted here | No | Derived participant metrics only |
| BETA EEG | Selection-aware evaluation and the released 15 checkpoints | NEMAR `nm000129`; Figshare `12264401`; Liu et al. 2020 | Non-commercial research use recorded by NEMAR | No raw EEG; BETA-trained weights only | Weights and documentation are CC BY-NC 4.0; users must also comply with BETA terms |
| Wearable SSVEP EEG | Frozen dry/wet transfer analysis | Figshare `13560281.v4`; Zhu et al. 2021 | Authoritative record terms govern reuse; no licence identifier asserted here | No | Derived participant metrics only |
| HarmonicFoldNet source | Proposed implementation and evaluation tooling | This repository | PolyForm Noncommercial 1.0.0 | Yes | Noncommercial source-available research software, not OSI open source |
| HarmonicFoldNet weights and documentation | Fifteen BETA fold checkpoints, model card and paper artifacts | Hugging Face release manifest | CC BY-NC 4.0 | Yes | Scope and exclusions are stated in the manifest and model card |
| SSVEPformer / FB-SSVEPformer adaptation | Local clean implementation based on the published method and public reproduction | Paper DOI `10.1016/j.neunet.2023.04.045`; public reproduction HEAD recorded as `fa21513054c8f8853d18d91b093f5c86d09f8e0b` on 2026-10-04 | No third-party source copied into the local implementation | No third-party source | Protocol adaptation, not bit-identical reproduction |
| MTSNet comparison | External protocol reconstruction | `https://github.com/lanzhen19/MTSNet`, commit `890b0a4f93c3affd50741a1f1d036fb74a30a062` | No licence file found in the audited commit | No | Upstream source is excluded; users obtain it separately under owner terms |

The weight licence does not relicense any dataset. The public model snapshot is
limited to 15 BETA source-decoder checkpoints. Benchmark, ablation, Wearable,
comparator and participant-adapter checkpoints are not represented as part of
that snapshot.

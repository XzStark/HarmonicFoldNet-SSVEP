# Literature and citation verification matrix

Status: working manuscript evidence map
Verification date: 2026-10-03
Scope: targeted narrative and novelty-oriented search, not a systematic review.

## Verification standard

The bibliography contains only records whose bibliographic identity was
checked against at least two of the following, where available: the publisher
or proceedings page, PubMed/PMC, Crossref, IEEE Xplore, and the current arXiv
record. DOI registration was checked separately from title, author, year, and
venue identity. A resolvable DOI alone was not treated as proof that the DOI
belonged to the intended work.

Searches were run on 2026-10-03 using combinations of `SSVEP`,
`cross-subject`, `calibration-free`, `Transformer`, `local-global`,
`time-frequency`, `harmonic`, `short-window`, `task attention`, `TRCA-R`,
`structural reparameterization`, `EEG`, `wearable`, and `deployment`.
Discovery used PubMed/PMC, Crossref, IEEE Xplore, arXiv, publisher pages, and
backward citation checking from the closest methods. The search was designed
to support manuscript writing and claim control; it does not establish
exhaustive coverage or a legal novelty opinion.

## Core datasets

| Key | Dataset and verified source | What the source establishes | Manuscript use and boundary |
| --- | --- | --- | --- |
| `Kim2025BetaRangeDataset` | Kim et al., *Scientific Data* 12, 1751 (2025), DOI `10.1038/s41597-025-06032-2` | Forty-class beta-range SSVEP speller dataset; 40 participants and 9,600 trials. | Development and beta-band stress-test source. It must not be described as untouched external validation after architecture selection. |
| `Wang2017BenchmarkDataset` | Wang et al., *IEEE TNSRE* 25, 1746--1752 (2017), DOI `10.1109/TNSRE.2016.2627556` | Canonical 35-participant, 40-target benchmark dataset and its acquisition/stimulus protocol. | Principal established laboratory benchmark; cite for participant count, target grid, sessions, sampling, and phase/frequency encoding. |
| `Liu2020BETADataset` | Liu et al., *Frontiers in Neuroscience* 14, 627 (2020), DOI `10.3389/fnins.2020.00627` | Seventy-participant BETA database acquired under less controlled conditions. | Independent multi-participant test bed; cite for participant count, four-block structure, varied trial duration, and environmental difficulty. |
| `Zhu2021WearableDataset` | Zhu et al., *Sensors* 21, 1256 (2021), DOI `10.3390/s21041256` | Paired wet/dry, eight-channel recordings from the same 102 participants in a 12-target wearable paradigm. | Supports electrode-condition analysis. Wet and dry recordings are paired conditions, not 204 independent participants; public data are not proof of performance on the intended product hardware. |

Together these sources substantiate four public datasets and 247 unique
participants (40 + 35 + 70 + 102). The count is a cohort count, not a claim
that all four datasets play identical experimental roles.

## Classical and calibrated baselines

| Key | Method/source | Verified methodological role | Claim-control note |
| --- | --- | --- | --- |
| `Lin2006CCA` | Standard CCA for SSVEP frequency recognition | Training-free correlation with sinusoidal harmonic references. | CCA and reference-signal matching are established priors, not a contribution. |
| `Chen2015FBCCA` | Filter-bank CCA | Harmonic sub-bands improve high-speed recognition over single-band CCA. | A filter bank plus harmonic evidence is established prior art. |
| `Nakanishi2018TRCA` | TRCA and ensemble TRCA | Learns participant-specific spatial filters by maximizing inter-trial reproducibility. | Report in the participant-calibrated track only; ensemble TRCA is a variant in the same source, not an independent zero-calibration comparator. |
| `Wong2020UnifiedSpatialFiltering` | Unified spatial-filtering framework; reference-signal-enhanced TRCA/eTRCA variants | Formalizes spatial filtering and reference-signal improvements, including the method often called TRCA-R in SSVEP toolboxes. | The paper must state whether its `TRCA-R` implementation follows this reference-signal construction. |
| `Lee2022RegularisedTRCA` | Regularised TRCA | A distinct robust regularisation approach also abbreviated TRCA-R in parts of the literature. | Do not cite this as the implemented comparator unless the covariance regularisation matches the local code. The abbreviation is ambiguous. |
| `Liu2021TDCA` | Task-discriminant component analysis | Individually calibrated spatiotemporal discriminant method evaluated on major public datasets. | Keep separate from calibration-free models and disclose the exact number of target-participant blocks. |

## Learned SSVEP decoders and closest method families

| Key | Primary overlap | Material distinction or protocol issue |
| --- | --- | --- |
| `Lawhern2018EEGNet` | Compact depthwise/separable convolutional EEG baseline. | General-purpose EEG architecture; protocol adaptation must be disclosed and should not be labelled an exact reproduction unless input and training contracts match the source. |
| `Ding2021FBtCNN` | Short-window time-domain CNN with filter-bank extension. | Strong short-window convolutional comparator; calibrated/source protocol differs from unseen-participant evaluation and must be reconciled. |
| `Chen2023SSVEPformer` | Transformer over complex spectral SSVEP features; the paper also reports a filter-bank variant. | Direct prior against novelty claims based on attention, complex FFT input, or generic harmonic-filter-bank Transformer fusion. Local runs are protocol-adapted reproductions. |
| `Wang2023TaskAttention` | Sine/cosine references and participant templates used as task-specific kernels in a compact attention-inspired network. | Direct prior against claiming that candidate references or task-conditioned attention are new. It is an individually calibrated method, unlike the zero-target-data main track. |
| `Huang2023CrossSubjectTransfer` | Cross-subject domain generalization without target-subject EEG. | Establishes calibration-free cross-subject transfer as an existing problem setting; it does not use the proposed folded local-to-global spectral-token design. |
| `Liu2024DGConformer` | Convolutional Transformer plus domain generalization for cross-subject SSVEP classification. | Close prior for local convolution plus global attention and unseen-subject generalization. Distinction must rely on candidate-aligned reduction, explicit harmonic bias, algebraic folding, and mechanism/deployment evidence rather than the generic hybrid topology. |
| `Lan2025MTSNet` | Multi-scale temporal and complex-spectral Convformer fusion. | Closest published prior for calibration-free temporal-spectral fusion. Same-protocol local comparison is required; published numbers are not interchangeable with local results. |
| `Li2025SSVEPPoolformer` | PoolFormer-style mixing and adaptive denoising. | Precludes novelty based solely on importing a vision-family token mixer or replacing self-attention with pooling. |
| `Yue2025AETF` | Spatial and learned frequency filtering followed by a full-sequence Transformer, with template/background mixing augmentation. | Precludes generic novelty for frequency filtering before temporal self-attention. It retains full temporal length rather than applying the proposed hierarchical token reduction. |
| `Dai2025TFFNet` | Parallel raw-time and complex-spectrum branches with learned fusion. | Precludes broad claims for temporal-spectral fusion or dynamic branch weighting. Its dual-branch topology differs from a shared candidate-token path. |
| `Huang2025IncepFormerNet` | Multi-scale temporal convolution, filter banks, and multi-head attention. | Peer-review status remains a preprint; cite as such. It narrows any claim based on generic multi-scale CNN-plus-Transformer composition. |
| `Wang2025CalibrationFreeDetection` | Short-window calibration-free decoding, distribution alignment, spectral denoising, and compact inference. | Published accuracies and timing use different source-subject counts, windows, batching, and hardware. Use only in a protocol-reconciliation/literature table unless reproduced. |
| `Ding2026JFPTS` | Joint frequency-phase training for deep SSVEP classification. | Direct prior against novelty claims for jointly exploiting frequency and phase. Its two-stage sampling/training strategy is distinct from a candidate-conditioned harmonic bias. |
| `Lan2025RMKD` | Long-window-to-short-window knowledge distillation. | Cross-window supervision is established prior art and should be a baseline or ablation, not part of the core novelty statement. |
| `Liu2020DMCCA` | Deep multiset CCA between EEG and reference signals. | Learned reference correlation is established; novelty cannot rest on neural correlation with sine/cosine references. |

## Structural reparameterization and deployment evidence

| Key | Source contribution | How it constrains this manuscript |
| --- | --- | --- |
| `Ding2021RepVGG` | Defines train-time multi-branch to inference-time plain-network structural reparameterization by algebraic kernel fusion. | Cite for the general design principle. The contribution here must be the verified one-dimensional EEG/candidate-token realization and its measured consequences, not structural reparameterization itself. |
| `Li2023RepNetMMCD` | Uses multi-scale depthwise EEG branches that are equivalently converted into a single convolution for seizure prediction. | This is the closest verified EEG prior and makes any “first structural reparameterization for EEG” statement false. It is not an SSVEP cross-subject Transformer and does not establish the complete proposed pipeline. |
| `Chen2015HighSpeedSpelling` | Joint frequency-phase coding, causal online spelling, visual latency handling, and ITR in an operational SSVEP BCI. | Supports the practical BCI and frequency/phase background, but does not validate the present model online. |
| `Wolpaw2002BCI` | Foundational BCI communication/control framing and rate-performance context. | Use sparingly for background and ITR context. It does not justify a device, clinical, or online-performance claim. |
| `Vialatte2010SSVEPReview` | Broad physiological and BCI review of SSVEP paradigms. | Supports general SSVEP background, not architecture novelty. |

The deployment claim must remain tied to batch-one measurements from the
folded graph on the reported desktop hardware. Parameter count is not a proxy
for speed, and desktop inference is not wearable-device validation.

## Citation placement plan

| Manuscript section | Minimum evidence set |
| --- | --- |
| Introduction: SSVEP/BCI motivation | `Vialatte2010SSVEPReview`, `Wolpaw2002BCI`, `Chen2015HighSpeedSpelling` |
| Dataset section | all four dataset keys above |
| Classical methods | `Lin2006CCA`, `Chen2015FBCCA`, `Nakanishi2018TRCA`, `Wong2020UnifiedSpatialFiltering`, `Liu2021TDCA` |
| Deep-learning related work | `Lawhern2018EEGNet`, `Ding2021FBtCNN`, `Chen2023SSVEPformer`, `Liu2024DGConformer`, `Lan2025MTSNet`, `Li2025SSVEPPoolformer`, `Yue2025AETF`, `Dai2025TFFNet` |
| Harmonic/candidate/frequency-phase context | `Wang2023TaskAttention`, `Liu2020DMCCA`, `Ding2026JFPTS`, `Chen2015HighSpeedSpelling` |
| Short-window/calibration-free context | `Ding2021FBtCNN`, `Wang2025CalibrationFreeDetection`, `Lan2025RMKD`, `Huang2023CrossSubjectTransfer` |
| Folding and deployment | `Ding2021RepVGG`, `Li2023RepNetMMCD` |
| Methods-process disclosure, if required | `Kassis2026ScientificAgentSkills` |

## Corrected citation errors found in the prior audit

| Prior value | Verified correction | Evidence |
| --- | --- | --- |
| PoolFormer paper DOI `10.1109/TCE.2025.3534427` | `10.1109/TCE.2025.3535157` | IEEE Xplore record 10855609 plus Crossref identity match. |
| Task-attention paper DOI `10.1109/TNSRE.2023.3275838` | `10.1109/TNSRE.2023.3276745` | IEEE paper/repository record plus Crossref identity match. |
| TDCA DOI `10.1109/TNSRE.2021.3076278` | `10.1109/TNSRE.2021.3114340` | IEEE/source manuscript plus Crossref identity match. |
| SSVEP review DOI `10.1016/j.pneurobio.2009.11.003` | `10.1016/j.pneurobio.2009.11.005` | PubMed PMID 19963032 plus Crossref identity match. |
| RepVGG DOI `10.1109/CVPR46437.2021.01328` | `10.1109/CVPR46437.2021.01352` | CVF proceedings page plus Crossref/DBLP identity match. |
| DOI `10.1088/1741-2560/12/4/046017` described as joint frequency-phase SSVEP work | This DOI belongs to an unrelated cerebral-palsy muscle-synergy article. Use `Chen2015HighSpeedSpelling` for the verified joint frequency-phase speller source. | Crossref title identity plus PubMed PMID 26483479. |

## Unresolved or deliberately excluded sources

The following items are not in `references.bib` and must not be cited as
verified publications until the stated issue is resolved:

1. **VIBE, ICLR 2026 submission.** An accessible OpenReview PDF exists, but the
   located record was a submission rather than a stable published version.
   Re-check title, author list, decision, and final version immediately before
   submission.
2. **Recent 2026 SSVEP preprints and very recent journal articles.** SQ-HAF,
   CSST/FBEA, and other late-breaking candidates should be refreshed at the
   final literature gate because publication status and metadata may change.
3. **Patent-only or abstract-only records.** RepEEG-Net and recent
   harmonic-attention/candidate-frequency Chinese patent records were retained
   in the separate novelty audit, not the scholarly BibTeX file. Abstract-only
   access is insufficient for method- or claim-level statements.
4. **Proprietary or inaccessible full text.** If a method claim is based only
   on an abstract, search excerpt, or secondary summary, restrict the statement
   to what that record explicitly says and mark full-text assessment as not
   completed.
5. **TRCA-R naming.** The project must map the implementation equations to
   either `Wong2020UnifiedSpatialFiltering` or `Lee2022RegularisedTRCA` before
   the Methods and comparison table are finalized. The abbreviation alone is
   not a sufficient citation identity.

## Required final checks after the manuscript exists

1. Validate every in-text key against `references.bib`; permit no unresolved
   keys and remove uncited entries.
2. Re-run DOI registration checks and manually compare title, first author,
   year, venue, volume, pages/article number, corrections, and retractions.
3. Re-check arXiv items for a later peer-reviewed version and replace the
   preprint when a version of record exists.
4. Audit every numerical or protocol claim against the actual source text,
   especially source-subject count, target calibration, channel montage,
   latency offset, window definition, and whether evaluation is online.
5. Keep published external numbers out of same-protocol ranking tables unless
   the manuscript labels the protocol mismatch explicitly.
6. Resolve the CVPR pagination discrepancy for `Ding2021RepVGG`: the official
   CVF proceedings page lists 13733--13742, whereas Crossref's IEEE deposit
   lists 13728--13737. The BibTeX currently follows the official CVF page.

## Process reference

The literature and citation workflow used here follows the multi-source search,
metadata-enrichment, and DOI-identity checks described by
`Kassis2026ScientificAgentSkills`. This process reference documents the method
used to assemble the bibliography; it is not evidence for any SSVEP claim.

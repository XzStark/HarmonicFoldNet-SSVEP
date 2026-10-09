# HarmonicFoldNet related-work and novelty audit

Status: working audit for experiment design. It is not a novelty opinion and
must not be converted into a claim of "first" until the cited methods and a
broader scholarly/patent search have been checked in full text.

## Search refresh: 2026-10-03

This scoping refresh used exact-name searches plus combinations of `SSVEP`,
`cross-subject`, `Transformer`, `local-global`, `structural
reparameterization`, `harmonic attention`, `candidate frequency`, `token
reduction`, `compact decoder` and `wearable`. Discovery covered PubMed,
Crossref, arXiv, publisher pages and publicly searchable Chinese patent
records. Venue fit was checked against current publisher scope pages. This is
a reproducible scoping search, not a systematic review, trademark clearance or
legal freedom-to-operate opinion.

The refresh found no meaningful scholarly/model collision for the exact name
`HarmonicFoldNet`. The name can therefore be retained, but the manuscript must
define **Fold** as train-to-deploy structural folding rather than a
cross-validation fold.

The closest method families remain SSVEPformer/FB-SSVEPformer, DG-Conformer,
MTSNet, IncepFormerNet, SSVEPPoolformer and recent candidate- or
harmonic-conditioned methods. Collectively, they make the following broad
claims unsafe: first Transformer for SSVEP, first local-plus-global EEG model,
first harmonic attention, first vision-family mixer in SSVEP, and first
reparameterized EEG network. The defensible contribution is narrower: the
specific candidate-aligned local-to-global topology, reduced-token late
harmonic-biased attention, numerically verified train-to-deploy folding,
single-weight multi-window behavior, direct mechanism evidence, and an
optional 113-parameter personalization track.

## Proposed contribution under test

The candidate contribution is not CNN plus attention, frequency-domain EEG
classification, CCA plus a neural network, or joint frequency/phase use. All
of those ideas have prior work.

The narrower method under test is an original one-dimensional local-to-global
hybrid Transformer for SSVEP decoding:

1. candidate-aligned local spectral/phase tokens are processed first by
   foldable depthwise-convolution local stages;
2. hierarchical token reduction precedes late global attention, so attention
   operates on a compact candidate representation rather than raw EEG samples;
3. training-time overparameterized local branches are folded into a
   numerically equivalent deployment graph; and
4. an optional regularized analytic sine/cosine and phase-demodulation score is
   used as a residual prior, not as the main architecture or novelty claim.

The model is evaluated causally at 0.4--1.2 s on unseen participants, with
deployment latency and graph-equivalence measurements. The frozen protocol-v4
evidence model remains a strong baseline. If the hybrid model does not improve
that baseline and survive component ablations, the local-to-global architectural
claim is unsupported and the model must not be presented as the paper's main
contribution.

## Closest located work

| Work | Overlap | Material distinction to test and disclose |
| --- | --- | --- |
| RepNet-MMCD (Li et al., IEEE TNSRE 2023, DOI 10.1109/TNSRE.2022.3217929) | Structural reparameterization and multi-scale depthwise convolution for lightweight EEG inference | This prior work folds parallel 3x3/5x5 depthwise branches for patient-specific seizure prediction and couples the network to uncertainty estimation. It precludes any claim that structural reparameterization is first used for EEG. It does not establish a candidate-aligned SSVEP hybrid Transformer, reduced-token late attention, harmonic guidance, unified multi-window cross-subject decoding, or our deployment/effect-mechanism evidence. |
| RepEEG-Net in CN122337088A (Shenzhen University, 2026) | A lightweight EEG network described with structural reparameterization and quantization | The available abstract places RepEEG-Net inside a brain-computer-interface teaching system using head-posture gating and complex-Morlet time-frequency images on an edge-AI hub. This is material prior art against broad claims such as "first reparameterized EEG network". The located disclosure does not show the complete SSVEP-specific local-to-global pipeline above. Claim-level review remains required because the CNIPA query endpoint failed during the 2026-10-02 audit and only the published abstract was retrievable. |
| Harmonic-alignment encoder in CN121116079B (Institute of Automation, Chinese Academy of Sciences, priority 2025-11-13) | Separately extracts fundamental and harmonic-band features with different convolution sizes/dilations, learns harmonic weighting, and applies self-attention to form a compressed EEG representation | This granted patent is close prior art against broad claims based on "harmonic branches plus self-attention" or separately dilated fundamental/harmonic convolution. Its independent claim additionally requires a modulation-enhancement model trained with pre-/post-neural-modulation data and a state discriminator, which the proposed model does not use. The paper distinction must remain candidate-conditioned attention over shared reduced spectral tokens, train-to-deploy local mixing and calibration-free multi-window evaluation; a new design must not simply recreate the patent's separate fundamental/harmonic convolution branches. |
| LLM-guided SSVEP interaction patent CN122756484A (2026) | Generates a dynamic flicker interface from scene candidates and decodes the resulting SSVEP with a disclosed "harmonic-attention fusion network" | Directly precludes treating harmonic attention for SSVEP as a first or sufficient novelty. The available official abstract does not establish the proposed local-first, reduced-token, train-to-deploy cross-subject decoder, but its full claims must be audited before any patent-facing harmonic-attention language is used. |
| Multimodal head-motion/SSVEP patent CN118519539A (2024) | Parallel Transformer extraction of time-domain and FFT-domain EEG features in a wearable character-input system | Precludes a generic novelty claim for parallel temporal/frequency Transformers or wearable SSVEP decoding. The patent is not a structural-reparameterized, candidate-conditioned reduced-token model, but its claims and description must be cited if dual-domain Transformer language is used. |
| Fast calibration-free SSVEP framework (Wang et al., arXiv:2506.01284, 2025) | Calibration-free short-window decoding, compact architecture and inference-efficiency claims on Benchmark, BETA and Nakanishi | Uses inter-trial remixing, context-aware distribution alignment, adaptive spectral denoising and a small convolution/MLP classifier. It reports LOSO training on the other 34/69 participants, eight channels, 0.3--0.7 s windows, 35.4% Benchmark accuracy at 0.4 s, and CPU timing on 240-sample batches. Our current fixed five-fold protocol retains a separate validation set and therefore uses fewer source participants; its numbers are not directly comparable. No public code link was present in the arXiv source audited on 2026-10-01. It directly precludes broad novelty claims based only on short windows, calibration-free evaluation or model efficiency. |
| JFPTS, *Journal of Neural Engineering* (2026), DOI 10.1088/1741-2552/ae36f6 | Joint use of SSVEP frequency and phase | A two-stage sampling/training strategy; not presently shown to use our analytic prior plus shared candidate-wise residual construction. Full text still requires method-level audit. |
| AETF (2025) | Spatial filtering, learned frequential convolution, and a multilayer Transformer for SSVEP, paired with template/background EEG augmentation | Precludes a generic claim for frequency filtering before Transformer decoding or for augmentation-driven cross-subject robustness. Its full-sequence Transformer explicitly retains all samples rather than using hierarchical token reduction and candidate-conditioned late attention. |
| Compact task-attention SSVEP network (Wang et al., IEEE TNSRE 2023) | Predefined sine/cosine and template kernels are interpreted as attention queries/keys; multi-head task attention and filter-bank variants learn spatial/feature weights | This is close prior art against calling candidate reference signals or task-specific attention intrinsically novel. It is a calibrated individual-recognition method using target-participant templates, whereas the proposed main track is zero-target-data cross-subject decoding. The supervision difference and candidate-conditioned harmonic-bias distinction must be explicit. |
| RMKD (Lan et al., *Neural Networks* 2025, DOI 10.1016/j.neunet.2025.107133) | Transfers long-window frequency features and logits to a short-window SSVEP decoder using masked feature generation, non-target-class distillation and inter-class-relation distillation | Precludes a novelty claim based on long-window-to-short-window distillation itself. Any use of cross-window supervision must be treated as a training baseline or must introduce and ablate a materially different mechanism. |
| SSVEP-TFFNet (2025) | Parallel raw-time and complex-spectrum branches with dynamic weighting for cross-subject SSVEP recognition | Directly precludes broad claims for dynamic time-frequency fusion or a reliability gate between temporal and spectral branches. The proposed mainline instead uses a compact spectral-token pipeline; a failed v4.4 gate experiment is retained only as a negative development result. |
| SQ-HAF (2026) | Few-shot cross-subject SSVEP recognition with source-quality/confidence gating, harmonic evidence and hierarchical adaptation | Precludes presenting confidence/reliability gating of harmonic SSVEP evidence as a stand-alone novelty. It also belongs to a different few-shot target-adaptation track and must not be mixed with zero-target-data results. |
| G-CMTF Net (2026) and related reliability-gated EEG fusion models | Reliability-aware gated temporal/spectral or cross-modal fusion using agreement, uncertainty or signal-quality evidence | Establish that reliability-aware evidence gating is a generic EEG fusion pattern, not a new mathematical principle. A gate can remain an engineering component only if it gives reproducible gains; it cannot carry the paper's novelty claim. |
| TSformer-SA (2024) | Temporal-spectral Transformer views, cross-view interaction and multi-view consistency learning for EEG decoding | Precludes a broad novelty claim for temporal-spectral consistency constraints. SSVEP-specific candidate conditioning and the deployable foldable-local-mixer topology would still need separate evidence. |
| FCPNet patent CN122757888A (2026) | Candidate-frequency-conditioned lightweight correlation | Builds class prototypes from training trials and candidate-conditioned spatial projections for high-frequency, limited-calibration decoding. Our current question is target-user-calibration-free analytic references and causal cross-subject short windows, not learned class-prototype matching. |
| Frequency/phase spatiotemporal beamforming (2016) | Explicit joint frequency-phase decoding | Participant-specific beamforming/thresholding rather than the proposed shared neural residual and unseen-participant protocol. |
| DMCCA (2020), DOI 10.1016/j.neucom.2019.10.049 | Learns nonlinear representations between EEG and sine/cosine references | Establishes that learned reference correlation is not itself new. The distinction must come from candidate-aligned residual design, causal short-window evaluation and deployment behavior. |
| SSVEPformer (2023), DOI 10.1016/j.neunet.2023.04.045 | Transformer processing of complex spectral features | Makes "attention for SSVEP" and "real/imaginary FFT input" unavailable as novelty claims. |
| FB-SSVEPformer (2023), same paper as SSVEPformer | Filter-bank harmonic information combined with a Transformer | Makes generic "harmonics plus Transformer" unavailable as a novelty claim. The proposed distinction must be the explicit candidate-conditioned harmonic bias, its mechanism analysis, the local-to-global reduced-token topology and deployment graph, not the mere use of harmonic information. |
| IncepFormerNet (Huang et al., arXiv:2502.13972, 2025) | Multi-scale temporal convolutions, multi-head attention and filter-bank SSVEP features | Precludes a generic "CNN plus Transformer plus multi-scale frequency information" claim. The disclosed architecture is Inception/filter-bank based; it is not reported as a train-to-deploy foldable local-mixing graph with candidate-aligned harmonic-biased reduced-token attention. |
| MTSNet (Lan et al., IEEE JBHI 2025, DOI 10.1109/JBHI.2025.3573410) | Calibration-free dual-branch temporal/complex-spectrum Convformer with multi-scale fusion on Benchmark and BETA | This is the closest current learned baseline to generic temporal-spectral fusion and must be compared. The authors publish a small PyTorch repository, but it contains no license file as audited on 2026-10-01; its source must remain a research reference and cannot be copied into the distributable project. A clean-room implementation or author permission is required for released reproduction code. |
| SSVEPPoolformer (Li et al., IEEE TCE 2025, DOI 10.1109/TCE.2025.3535157) | PoolFormer-style token mixing and adaptive denoising for SSVEP classification | Precludes novelty based only on importing a vision-family mixer into SSVEP. It does not by itself disclose our structural reparameterization, reduced-token late attention, explicit harmonic bias and multi-window deployment combination. |
| VIBE (ICLR 2026 submission) | Vision Transformer based experts for SSVEP and explicit local/global motivation | A close contemporary preprint against any "first vision Transformer for SSVEP" claim. Its ViT generation and expert specialization route is materially different, but its full protocol and publication status must be monitored before submission. |
| SSVEP-TFFNet (2025) | Time-frequency deep feature fusion and cross-subject motivation | Makes generic time/frequency fusion unavailable as a novelty claim. |
| XR wearable SSVEP-TFFNet evaluation (2026), DOI 10.3390/s26165102 | Subject-independent deep SSVEP decoding, XR stimulation and reduced-channel wearable evaluation | Makes generic "deep SSVEP for wearable/XR" unavailable as a novelty claim. It evaluates TFFNet and FBCCA on 30 XR users; it does not by itself establish the proposed analytic-prior/residual construction or our sub-second deployment claim. |
| CSST/FBEA (2026), arXiv:2601.21203 | Strong short-window cross-subject results | Uses unlabeled target-domain data for domain adaptation/self-training. It belongs to a different supervision track from a zero-target-data model and must not be compared as if supervision were identical. |

## Claims that are currently prohibited

- "the first CNN-attention SSVEP model";
- "the first use of structural reparameterization for EEG";
- "the first vision backbone or vision Transformer used for SSVEP";
- "the first model to combine frequency and phase";
- "the first neural CCA/reference-signal model";
- "the first harmonic-guided or frequency-prior attention model for SSVEP";
- "the first time-frequency Transformer for wearable SSVEP";
- "the first reliability-aware temporal-spectral gate for EEG or SSVEP";
- "the first long-window-to-short-window SSVEP knowledge-distillation method";
- "state of the art" before same-protocol strong baselines are complete;
- "works on wearable glasses" before measurements on the intended electrode
  placement and acquisition hardware;
- any silent-speech, imagined-speech or thought-reading claim.

## Evidence gate before paper drafting

- complete fixed five-fold subject-disjoint out-of-fold evaluation for three
  training seeds on all five datasets;
- compare FBCCA, EEGNet, an explicitly labelled SSVEPformer reproduction and
  a clean-room MTSNet reproduction (or another licensed modern lightweight
  network) under identical channels, windows and participant splits;
- report calibrated TRCA/eTRCA/TRCA-R and TDCA in a separate,
  protocol-matched supervision track; never mix their published or reproduced
  calibrated numbers with the zero-target-data table;
- audit and, where reproducible code permits, compare the 2025 fast
  calibration-free SSVEP framework; do not substitute its published numbers
  for a same-protocol run;
- reproduce FB-SSVEPformer under the identical channel/window/split contract,
  or place its published result only in a separately labelled literature table;
- include a protocol-reconciliation table for the 2025 framework (LOSO source
  count, channel names, preprocessing, windows, batch definition and hardware)
  before discussing its published accuracy or latency beside our results;
- full HarmonicFold hybrid, no-evidence-prior, no-late-attention,
  no-foldable-local-convolution, local-only and training-versus-folded-graph
  ablations on the same frozen folds;
- participant-level confidence intervals, paired tests and multiplicity
  correction;
- final-model batch-one CPU/GPU latency, memory, parameter count and numerical
  train/deploy graph equivalence;
- full-text method audit of JFPTS and other candidate/reference-conditioned
  SSVEP models before using "novel" in the title or abstract.
- obtain and review the claims of CN122337088A before making any patent-facing
  structural-reparameterization claim; the abstract alone is not a freedom-to-
  operate or claim-scope opinion.
- keep any short-window extension distinct from CN121116079B: do not use its
  separately dilated fundamental/harmonic convolution branches as the claimed
  novelty; audit the final architecture claim-by-claim before submission.
- do not promote a reliability gate to a contribution unless it beats the
  frozen v4.1 model on the registered development fold and survives a
  gate-only/subband-only ablation. The first v4.4 combined candidate failed
  this gate (0.4/0.8/1.2 s: 28.571/67.202/77.798% versus the same-fold v4.1
  result 29.940/67.321/79.286%), so it must not replace the mainline.
- audit CN122756484A claim-by-claim before using "harmonic attention" in a
  patent-facing independent claim; the official abstract already creates a
  material near-neighbour.

## Sources

- Search and critical-appraisal workflow: Kassis T, Agarwal V, He Y, Patel D,
  Brueckner AM. *Scientific Agent Skills: A Library of Procedural Knowledge for
  Research Agents*. arXiv:2609.00065v2 (2026).
- JFPTS: https://pubmed.ncbi.nlm.nih.gov/41525762/
- RMKD: https://pubmed.ncbi.nlm.nih.gov/39862529/
- AETF: https://pmc.ncbi.nlm.nih.gov/articles/PMC12501431/
- G-CMTF Net: https://www.mdpi.com/2073-8994/18/2/316
- TSformer-SA: https://arxiv.org/abs/2401.06340
- SQ-HAF: https://www.mdpi.com/1424-8220/26/18/5830
- Compact task-attention network: https://doi.org/10.1109/TNSRE.2023.3276745
- RepNet-MMCD: https://pubmed.ncbi.nlm.nih.gov/36306304/
- RepEEG-Net patent CN122337088A (abstract/search record):
  https://eureka.patsnap.com/latest-cn-patents-106337
- Harmonic-alignment encoder patent CN121116079B:
  https://patents.google.com/patent/CN121116079B/zh
- LLM-guided SSVEP interaction patent CN122756484A (official CNIPA record):
  http://epub.cnipa.gov.cn/patent/CN122756484A
- Multimodal head-motion/SSVEP patent CN118519539A:
  https://patents.google.com/patent/CN118519539A/zh
- Fast calibration-free SSVEP framework: https://arxiv.org/abs/2506.01284
- FCPNet CN122757888A: https://eureka.patsnap.com/patent/CN122757888A
- Frequency/phase beamforming: https://pmc.ncbi.nlm.nih.gov/articles/PMC4972379/
- DMCCA: https://doi.org/10.1016/j.neucom.2019.10.049
- SSVEPformer: https://doi.org/10.1016/j.neunet.2023.04.045
- IncepFormerNet: https://arxiv.org/abs/2502.13972
- MTSNet: https://doi.org/10.1109/JBHI.2025.3573410
- MTSNet author code (no repository license observed on 2026-10-01):
  https://github.com/lanzhen19/MTSNet
- SSVEP-TFFNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC12515880/
- SSVEPPoolformer: https://doi.org/10.1109/TCE.2025.3535157
- VIBE submission: https://openreview.net/pdf?id=TND5hiXjeM
- XR wearable SSVEP evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC13517239/
- CSST/FBEA: https://arxiv.org/abs/2601.21203

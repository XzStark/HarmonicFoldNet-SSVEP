# Stage-2 draft quality report

Date: 2026-10-03

## Completed checks

- The manuscript is a complete English IMRaD draft with the structured
  Objective/Approach/Main results/Significance abstract required by the working
  venue.
- Twenty-seven in-text citation keys resolve to `references.bib`; no unresolved
  key remains.
- Bibliographic identities were checked against publisher, Crossref, PubMed,
  IEEE, or current arXiv records as available. The verification notes and
  corrected metadata are recorded in `LITERATURE_MATRIX.md`.
- The paper tree contains no deprecated model/project-name strings specified
  by the author.
- The draft avoids promotional claims, universal state-of-the-art language,
  device claims without device measurements, and any suggestion that public
  data were self-collected.
- A stop-slop prose pass found no canned openers, marketing phrases,
  manufactured rhetorical questions, fake quotations, or decorative
  conclusions. Evidence-based negative claim boundaries were retained because
  they prevent scientific overstatement.
- Six figures were regenerated from separate source tables. Every figure is
  available as editable PDF/SVG and 600-dpi PNG.
- The plotting source passed the publication-figure static preflight: 18 PASS,
  0 FAIL, and 3 non-blocking warnings (TIFF not exported, final journal width
  to reconfirm, and a conservative log-axis guard warning despite an explicit
  positivity check).
- All six rendered PDFs passed the collision audit. All PDF text runs are at
  least 5.5 pt after the final correction. Multi-panel Figures 2-6 passed the
  render-time alignment gate.
- Visual inspection was completed for every PNG after the automated audits.
- The manuscript follows the deployment table rather than an inconsistent
  prose sentence in the frozen summary: the filter-bank reference is slightly
  faster in CPU median latency, while HarmonicFoldNet is much faster on the
  measured GPU and uses less memory.
- The preserved `runs` tree was recovered and audited. Main Benchmark/BETA
  groups contain all registered participants and three seeds; recomputed
  balanced-accuracy means match the frozen comparison artifacts to numerical
  tolerance.
- Participant-level ITR is restored from the original per-participant run
  metrics. It is calculated before group aggregation and reported with 10,000-
  resample participant-bootstrap confidence intervals.
- The final no-local, no-attention and no-harmonic-bias ablations now have
  participant-level source data, paired difference confidence intervals,
  rank-biserial effects and the registered within-ablation Holm correction.
- The MTSNet family was re-audited with one Holm correction spanning both
  datasets and all three registered windows (six tests). The substantive
  conclusion is unchanged: HarmonicFoldNet is lower at 0.4 s and statistically
  tied at 0.8 and 1.2 s.
- Figures 2 and 3 were regenerated from the restored participant-level
  evidence. Figure 2 now shows participant-bootstrap confidence bands and
  Figure 3c shows paired-difference confidence intervals. All six rendered
  figures pass the collision audit after this update.

## Evidence limits that remain visible

1. The existing EEGNet package does not have the same three-seed depth as the
   principal comparisons. It is not used as a headline result.
2. Authors, affiliations, funding, CRediT roles, competing-interest wording,
   and the final venue-compliant AI-use statement remain author/submission
   fields.

## Tool limitation

The academic-paper skill's documented acronym-check script is not present in
the installed skill directory. A repository-local deterministic acronym audit
will therefore be used in the final submission QA and identified as a local
check rather than the missing skill script.

## Stage boundary

This report completes drafting self-review only. Formal evidence-integrity
verification and the independent multi-perspective peer review belong to the
next pipeline stages.

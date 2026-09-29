# v0.1 release checklist

## Completed locally

- [x] Fixed subject-disjoint result snapshot
- [x] Training and deployment checkpoints
- [x] Numerical reparameterization equivalence test
- [x] Evaluation JSON with per-participant results
- [x] Noncommercial source-code license
- [x] Separate noncommercial model/documentation license
- [x] Citation metadata
- [x] Technical report source
- [x] Hugging Face model card source
- [x] Public comparison-figure prompt
- [x] Consistent progress snippets for external materials
- [x] SHA-256 checksums for the frozen upload bundle
- [x] Secret and local-path scan of release-facing files
- [x] Confirmed no raw EEG or self-recorded biometric data is Git-tracked

## Required immediately before public release

- [x] Add the final Git repository URL
- [x] Add the final Hugging Face model URL
- [ ] Create Git tag and release `v0.1.0`
- [ ] Archive the release with Zenodo and add the DOI
- [ ] Add the DOI and URLs to `CITATION.cff`, README, report, and model card
- [ ] Confirm the author's preferred public email or omit email everywhere
- [ ] Verify all tests from a clean environment
- [ ] Confirm target venue policy before submitting the report as a preprint

## Language and claim controls

- Do not claim state of the art.
- Do not describe model compute latency as end-to-end interaction latency.
- Always state that the current result uses a five-second window and one fixed
  participant split.
- Keep external paper numbers separate unless protocol, dataset, channels,
  preprocessing, and split are reproduced consistently.
- Describe the release as noncommercial source-available research software, not
  OSI-approved open source.

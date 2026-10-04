from scripts.audit_manuscript import audit_acronyms, audit_citations, audit_text


def test_citation_audit_detects_missing_and_uncited_keys() -> None:
    findings = audit_citations("Claim [@Present; @Missing].", "@article{Present,}\n@article{Unused,}\n")
    assert [(item.code, item.message) for item in findings] == [
        ("missing-bibliography-key", "Missing"),
        ("uncited-bibliography-entry", "Unused"),
    ]


def test_acronym_audit_flags_use_before_definition() -> None:
    findings = audit_acronyms("# Title\n\nBCI is studied. Brain-computer interface (BCI) decoding follows.")
    assert any(item.code == "acronym-before-definition" and item.message.startswith("BCI") for item in findings)


def test_text_audit_flags_prohibited_names_and_placeholders() -> None:
    legacy_name = "Fast" + "VIT"
    findings = audit_text(f"A legacy {legacy_name} reference. Authors: [to be completed]")
    assert {item.code for item in findings} == {"prohibited-name", "author-placeholder"}


def test_text_and_acronym_audits_ignore_repository_url_identifiers() -> None:
    legacy_url = "https://example.org/Fast" + "SSVEPFusionNet"
    text_findings = audit_text(f"Code: {legacy_url}")
    acronym_findings = audit_acronyms("Code: https://example.org/KSTARKX/HarmonicFoldNet")
    assert not text_findings
    assert not acronym_findings

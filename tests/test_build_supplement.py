from pathlib import Path

from scripts import build_supplement


def test_supplement_uses_registered_evidence() -> None:
    build_supplement.main()
    output = Path("paper/SUPPLEMENTARY_INFORMATION.md").read_text(encoding="utf-8")
    assert "Benchmark windows of 2.0, 3.0 and 5.0 s" in output
    assert "MTSNet six-test multiplicity family" in output
    assert "no_harmonic_bias" in output
    assert "SHA-256" in output

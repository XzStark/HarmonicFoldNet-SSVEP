from scripts.audit_ai_style import heuristic_audit


def test_heuristic_flags_stock_language_without_claiming_authorship() -> None:
    result = heuristic_audit(
        "It is important to note that this highlights a key finding. "
        "Measured latency was 3.0 ms on the registered host."
    )
    assert len(result["stock_phrase_hits"]) >= 2
    assert "not a probability" in result["interpretation"]

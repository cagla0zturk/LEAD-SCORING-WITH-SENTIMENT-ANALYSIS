"""Tests for the per-lead score explainer (LightGBM SHAP contributions)."""

from __future__ import annotations


def test_explainer_returns_ranked_factors(trained_artifacts):
    from lead_priority.scoring.explain import ScoreExplainer

    explainer = ScoreExplainer.load()
    features = {
        "Lead Origin": "Landing Page Submission",
        "Lead Source": "Google",
        "TotalVisits": 8,
        "Total Time Spent on Website": 1800,
        "Page Views Per Visit": 4.0,
        "What is your current occupation": "Working Professional",
        "Do Not Email": "No",
        "Do Not Call": "No",
    }
    factors = explainer.explain(features, top_k=4)
    assert 1 <= len(factors) <= 4
    # Ranked by absolute contribution (descending).
    contribs = [abs(f["contribution"]) for f in factors]
    assert contribs == sorted(contribs, reverse=True)
    for f in factors:
        assert f["label"]
        assert f["direction"] in {"artırıyor", "azaltıyor"}

"""End-to-end API tests using FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_artifacts):
    # Import after artifacts exist so lifespan loads real models.
    from lead_priority.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_score_positive_engagement(client):
    payload = {
        "features": {
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Google",
            "TotalVisits": 6,
            "Total Time Spent on Website": 1500,
            "Page Views Per Visit": 3.5,
            "Last Activity": "SMS Sent",
            "What is your current occupation": "Working Professional",
            "Do Not Email": "No",
            "Do Not Call": "No",
        },
        "interaction_text": "Müşteri çok ilgili, demo talep etti, hemen başlamak istiyor.",
    }
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["conversion_probability"] <= 1.0
    assert body["conversion_prediction"] in (0, 1)
    assert body["sentiment"]["label"] == "positive_engagement"
    assert 0.0 <= body["priority"]["priority_score"] <= 1.0
    assert body["priority"]["tier"] in {"hot", "warm", "cooling", "cold"}
    # Segmentation enrichment is present.
    seg = body["segmentation"]
    assert seg["action_bucket"] in {"call_today", "rescue_email", "nurture", "lost"}
    assert seg["playbook"]["email_subject"]
    assert seg["insights"]["recommended_channel"]


def test_score_empty_interaction_defaults_neutral(client):
    resp = client.post("/score", json={"features": {"TotalVisits": 1}})
    assert resp.status_code == 200
    assert resp.json()["sentiment"]["label"] == "neutral"


def test_top_leads_sorted(client):
    resp = client.get("/leads/top?n=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] <= 5
    scores = [lead["priority_score"] for lead in body["leads"]]
    assert scores == sorted(scores, reverse=True)


def test_morning_brief(client):
    resp = client.get("/dashboard/brief?n_call=5&n_cooling=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_leads"] >= 1
    # "call today" is sorted by priority and only contains reachable leads.
    call = body["call_today"]
    assert [c["priority_score"] for c in call] == sorted(
        [c["priority_score"] for c in call], reverse=True
    )
    assert all(c["reachable"] for c in call)
    # Every "cooling" lead is flagged at-risk.
    assert all(c["is_cooling"] for c in body["cooling"])


def test_dashboard_segments_json(client):
    resp = client.get("/dashboard/segments")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["buckets"].keys()) == {"call_today", "rescue_email", "nurture", "lost"}
    assert data["total_leads"] >= 1
    # Every lead is assigned to exactly one bucket.
    assert sum(data["counts"].values()) == data["total_leads"]


def test_dashboard_html_renders(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Satış Dashboard" in resp.text


def test_root_redirects_to_dashboard(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 308)
    assert resp.headers["location"].endswith("/dashboard")


def test_conversion_weight_override_changes_priority(client):
    payload = {
        "features": {"TotalVisits": 5, "Total Time Spent on Website": 1000},
        "interaction_text": "Telefonu açmadı, ilgisiz görünüyor.",
        "conversion_weight": 1.0,
    }
    high_conv = client.post("/score", json=payload).json()
    payload["conversion_weight"] = 0.0
    high_sent = client.post("/score", json=payload).json()
    # Disengaged text -> sentiment-weighted priority should be lower.
    assert high_sent["priority"]["priority_score"] <= high_conv["priority"]["priority_score"]

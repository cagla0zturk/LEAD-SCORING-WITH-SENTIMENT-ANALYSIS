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

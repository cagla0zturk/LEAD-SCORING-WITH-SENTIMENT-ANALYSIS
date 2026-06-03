"""Tests for rule-based segmentation, personas and playbooks."""

from __future__ import annotations

from lead_priority.segmentation import (
    ACTION_BUCKETS,
    LEAD_CATEGORIES,
    action_bucket,
    approach_insights,
    build_playbook,
    lead_category,
    objection_reason,
)


def test_objection_reason_keywords():
    assert objection_reason("Fiyatı çok yüksek, indirim var mı?") == "price"
    assert objection_reason("Şu an müsait değilim, sonra bakarım") == "timing"
    assert objection_reason("Başka bir firmayla karşılaştırıyoruz") == "competitor"
    assert objection_reason("Yöneticimin onayı gerekiyor") == "authority"
    assert objection_reason("") == "generic"


def test_persona_assignment():
    assert lead_category(0.9, "positive_engagement") == "ready_to_buy"
    assert lead_category(0.8, "objection", objection="price") == "price_objection"
    assert lead_category(0.8, "objection", objection="timing") == "timing_objection"
    assert lead_category(0.7, "disengaged") == "going_cold"
    assert lead_category(0.1, "disengaged") == "low_intent"
    cat = lead_category(0.6, "neutral", features={"Total Time Spent on Website": 1500})
    assert cat == "information_seeker"


def test_action_buckets_are_known():
    # going_cold -> rescue; ready_to_buy high -> call; unreachable -> lost
    assert action_bucket(0.8, "disengaged", reachable=True, category="going_cold") == "rescue_email"
    assert action_bucket(0.8, "positive_engagement", reachable=True, category="ready_to_buy") == "call_today"
    assert action_bucket(0.9, "positive_engagement", reachable=False, category="ready_to_buy") == "lost"
    assert action_bucket(0.1, "disengaged", reachable=True, category="low_intent") == "lost"
    for fn_result in ("call_today", "rescue_email", "nurture", "lost"):
        assert fn_result in ACTION_BUCKETS


def test_insights_respect_contact_prefs():
    ins = approach_insights({"Do Not Call": "Yes"}, "neutral")
    assert ins.recommended_channel == "e-posta"
    ins2 = approach_insights({"Do Not Email": "Yes"}, "neutral")
    assert ins2.recommended_channel == "telefon"


def test_playbook_is_tailored():
    feats = {"Specialization": "Finance Management", "What is your current occupation": "Working Professional"}
    ins = approach_insights(feats, "objection", objection="price")
    pb = build_playbook("price_objection", feats, ins)
    assert pb.category == "price_objection"
    assert pb.email_subject and pb.email_body and pb.call_opening
    assert "Finance Management" in pb.email_subject
    assert pb.category in LEAD_CATEGORIES

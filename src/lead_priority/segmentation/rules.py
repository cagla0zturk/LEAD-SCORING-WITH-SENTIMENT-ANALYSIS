"""Rule-based segmentation: action buckets, personas and approach insights.

Why rules (and not another ML model)?
-------------------------------------
A sales rep must understand *why* a lead is in a bucket and *how* to approach it. A
transparent, tunable rule layer on top of the calibrated probability + sentiment is far
more useful (and trustworthy) here than an opaque classifier, and product can adjust the
bands without retraining. The numbers feeding it (conversion probability, sentiment) are
the learned parts; this layer turns them into an action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

# --------------------------------------------------------------------------------------
# Action buckets — the four columns of the morning dashboard.
# --------------------------------------------------------------------------------------
ACTION_BUCKETS: Final[dict[str, str]] = {
    "call_today": "Bugün Ara",
    "rescue_email": "Mail ile Kurtar",
    "nurture": "İzle / Besle",
    "lost": "Kopanlar",
}

# Lead categories (personas) — drive the playbook (email + talk-track).
LEAD_CATEGORIES: Final[dict[str, str]] = {
    "ready_to_buy": "Satın Almaya Hazır",
    "price_objection": "Fiyat İtirazı",
    "timing_objection": "Zamanlama İtirazı",
    "competitor_objection": "Rakip Değerlendiriyor",
    "information_seeker": "Bilgi Topluyor",
    "going_cold": "Soğuyor (Kurtarılabilir)",
    "low_intent": "Düşük Niyet",
    "needs_nurturing": "Isıtılmalı",
}

# Bands (kept here so product can tune them in one place).
_HIGH_PROB: Final[float] = 0.60
_MED_PROB: Final[float] = 0.45
_LOW_PROB: Final[float] = 0.30
_HIGH_TIME_ON_SITE: Final[float] = 600.0  # seconds; "has really browsed"

# --------------------------------------------------------------------------------------
# Objection reason (refines the generic "objection" sentiment for the playbook).
# Keyword based, TR + EN. Used ONLY for playbook routing, never for model training.
# --------------------------------------------------------------------------------------
_OBJECTION_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "price": (
        "fiyat", "pahalı", "bütçe", "indirim", "ücret", "maliyet", "öde",
        "price", "expensive", "budget", "discount", "cost", "cheaper", "afford",
    ),
    "timing": (
        "zaman", "şimdi değil", "sonra", "müsait değil", "yoğun", "ileri tarih", "beklet",
        "timing", "later", "not now", "busy", "next quarter", "postpone", "schedule",
    ),
    "competitor": (
        "rakip", "başka firma", "başka bir firma", "firmay", "alternatif", "diğer",
        "karşılaştır", "kıyasl", "competitor", "another vendor", "compare", "alternative",
    ),
    "authority": (
        "yönetici", "onay", "patron", "karar verici", "ekip", "manager", "approval",
        "boss", "decision maker", "sign off",
    ),
}


def objection_reason(text: str | None) -> str:
    """Classify the *reason* behind an objection from free text (TR/EN keywords)."""
    if not text:
        return "generic"
    lowered = text.lower()
    # Priority order: price > timing > competitor > authority.
    for reason in ("price", "timing", "competitor", "authority"):
        if any(kw in lowered for kw in _OBJECTION_KEYWORDS[reason]):
            return reason
    return "generic"


# --------------------------------------------------------------------------------------
# Persona (lead category)
# --------------------------------------------------------------------------------------
def lead_category(
    conversion_probability: float,
    sentiment_label: str,
    *,
    features: dict[str, Any] | None = None,
    objection: str = "generic",
) -> str:
    """Assign an actionable persona from probability + sentiment (+ light features)."""
    features = features or {}
    p = conversion_probability

    if sentiment_label == "objection":
        if objection == "price":
            return "price_objection"
        if objection == "timing":
            return "timing_objection"
        if objection == "competitor":
            return "competitor_objection"
        # authority/generic objection on a valuable lead is still essentially price/value.
        return "price_objection" if p >= _MED_PROB else "needs_nurturing"

    if sentiment_label == "positive_engagement":
        return "ready_to_buy" if p >= _MED_PROB else "needs_nurturing"

    if sentiment_label == "disengaged":
        return "going_cold" if p >= _MED_PROB else "low_intent"

    # neutral
    time_on_site = _as_float(features.get("Total Time Spent on Website"))
    if time_on_site is not None and time_on_site >= _HIGH_TIME_ON_SITE:
        return "information_seeker"
    return "needs_nurturing" if p >= _LOW_PROB else "low_intent"


# --------------------------------------------------------------------------------------
# Action bucket (dashboard column)
# --------------------------------------------------------------------------------------
def action_bucket(
    conversion_probability: float,
    sentiment_label: str,
    *,
    reachable: bool,
    category: str,
) -> str:
    """Decide which dashboard column a lead belongs in.

    Mutually exclusive, evaluated in priority order:
      lost -> rescue_email -> call_today -> nurture
    """
    p = conversion_probability

    # Cannot contact at all, or low-value and gone cold -> drop / long-term list.
    if not reachable:
        return "lost"
    if category == "low_intent" and sentiment_label == "disengaged" and p < _LOW_PROB:
        return "lost"

    # Valuable but slipping away -> a tailored email is the right, low-pressure save.
    if category == "going_cold":
        return "rescue_email"

    # High-value + an engaged/objecting signal -> a live call beats an email.
    if p >= _MED_PROB and sentiment_label in {"positive_engagement", "objection", "neutral"}:
        return "call_today"

    # Everything else: keep it warm with light touches.
    return "nurture"


# --------------------------------------------------------------------------------------
# Approach insights — "hangi konudan yaklaşmalı, neyi sever/sevmez"
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ApproachInsights:
    """What to lead with, what to avoid, and which channel to use."""

    interests: list[str] = field(default_factory=list)  # ilgisini çekebilecek noktalar
    cautions: list[str] = field(default_factory=list)  # dikkat / sevmediği noktalar
    talking_points: list[str] = field(default_factory=list)  # konuşma açılış noktaları
    recommended_channel: str = "telefon"

    def as_dict(self) -> dict[str, Any]:
        return {
            "interests": self.interests,
            "cautions": self.cautions,
            "talking_points": self.talking_points,
            "recommended_channel": self.recommended_channel,
        }


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def approach_insights(
    features: dict[str, Any],
    sentiment_label: str,
    *,
    objection: str = "generic",
) -> ApproachInsights:
    """Derive human-readable approach guidance from the lead's features + sentiment."""
    interests: list[str] = []
    cautions: list[str] = []
    talking_points: list[str] = []

    occupation = str(features.get("What is your current occupation") or "").strip()
    specialization = str(features.get("Specialization") or "").strip()
    source = str(features.get("Lead Source") or "").strip()
    time_on_site = _as_float(features.get("Total Time Spent on Website"))
    dne = str(features.get("Do Not Email", "No")).strip().lower() == "yes"
    dnc = str(features.get("Do Not Call", "No")).strip().lower() == "yes"

    if time_on_site is not None and time_on_site >= _HIGH_TIME_ON_SITE:
        interests.append(
            f"Sitede ~{int(time_on_site/60)} dk geçirmiş; içeriği incelemiş, somut müfredat/özelliklerle ilerleyin."
        )
        talking_points.append("İncelediği konulara dair somut örnek ve müfredat detayı verin.")
    if occupation == "Working Professional":
        interests.append("Çalışan profesyonel: kariyer ilerlemesi, sertifika ve ROI vurgusu.")
        talking_points.append("Sertifikanın kariyere/terfiye etkisini öne çıkarın.")
    elif occupation in {"Student", "Unemployed"}:
        interests.append("Öğrenci/iş arayan: istihdam çıktısı ve esnek ödeme planı önemli.")
        talking_points.append("Mezuniyet sonrası iş/istihdam başarı oranlarını paylaşın.")
    if specialization:
        interests.append(f"İlgi alanı: {specialization} — buna özel içerik/örnek sunun.")
    if source in {"Reference", "Referral Sites", "Welingak Website"}:
        interests.append("Referansla geldi: güven yüksek; referans/başarı hikayeleriyle ilerleyin.")

    if sentiment_label == "objection":
        if objection == "price":
            cautions.append("Fiyat hassas: önce değer/ROI, sonra indirim veya ödeme planı.")
            talking_points.append("Ödeme planı ve fiyat-değer karşılaştırması hazırlayın.")
        elif objection == "timing":
            cautions.append("Zamanlama itirazı: baskı yapmayın; esnek başlangıç tarihi sunun.")
            talking_points.append("Esnek başlangıç ve takvim hatırlatması önerin.")
        elif objection == "competitor":
            cautions.append("Rakip değerlendiriyor: fiyat savaşına girmeden farklılaştırın.")
            talking_points.append("Bizi ayrıştıran 2-3 net faydayı vurgulayın.")
        else:
            cautions.append("İtiraz var: önce dinleyin, itirazın kök nedenini netleştirin.")
    elif sentiment_label == "disengaged":
        cautions.append("Mesafeli: aşırı temastan kaçının; tek, değer odaklı kişisel mesaj.")
        talking_points.append("Kısa, kişisel ve değer içeren tek bir dokunuş yapın.")
    elif sentiment_label == "positive_engagement":
        talking_points.append("İlgili: momentumu kaçırmadan net bir sonraki adım (demo/teklif) önerin.")

    # Channel recommendation respects contact preferences.
    if dnc and not dne:
        recommended_channel = "e-posta"
        cautions.append("Telefon istemiyor (Do Not Call): e-posta ile ilerleyin.")
    elif dne and not dnc:
        recommended_channel = "telefon"
        cautions.append("E-posta istemiyor (Do Not Email): telefonla ilerleyin.")
    elif dne and dnc:
        recommended_channel = "ulaşılamıyor"
        cautions.append("Hem telefon hem e-posta kapalı: aktif temas mümkün değil.")
    else:
        recommended_channel = "telefon" if sentiment_label != "disengaged" else "e-posta"

    if not interests:
        interests.append("Belirgin sinyal az; keşif soruları ile ihtiyacı netleştirin.")
    return ApproachInsights(
        interests=interests,
        cautions=cautions,
        talking_points=talking_points,
        recommended_channel=recommended_channel,
    )

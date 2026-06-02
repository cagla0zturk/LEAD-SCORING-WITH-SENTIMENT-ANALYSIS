"""Synthetic engagement-note generator (Turkish + English, 4 intent classes).

Why synthetic?
--------------
The X Education tabular dataset has **no free-text interaction field**, so there is
nothing real to run sentiment on. We therefore *generate* short CRM-style notes
("müşteri fiyatı yüksek buldu, demo istedi", "not interested, please stop calling")
from hand-written templates, one per intent class.

Leakage policy (critical - see README)
--------------------------------------
The sentiment training label is the *template class* that produced the text. That
label is drawn **independently of the tabular ``Converted`` target**. Consequences:

* Training the sentiment model is leakage-free: it only ever learns ``text -> intent``.
* When we attach a note to a lead for the combined-score *demo*, we again pick the
  intent independently of ``Converted`` by default (``conversion_aware=False``), so the
  combined score is not silently fed the answer. ``conversion_aware=True`` exists only
  to make the demo dashboard look realistic and is never used for model training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

import pandas as pd

from lead_priority.config import RANDOM_SEED, SENTIMENT_LABELS

# --------------------------------------------------------------------------------------
# Templates.  {x} placeholders are filled from the fillers below to add lexical variety.
# Each list mixes formal and informal phrasing the way real rep notes / customer emails do.
# --------------------------------------------------------------------------------------
_PRODUCTS: Final[list[str]] = ["kurs", "program", "eğitim", "course", "bootcamp", "program"]
_FEATURES: Final[list[str]] = [
    "sertifika",
    "ödeme planı",
    "demo",
    "müfredat",
    "certificate",
    "payment plan",
    "syllabus",
    "mentorluk",
]

_TEMPLATES: Final[dict[str, list[str]]] = {
    "positive_engagement": [
        "Müşteri çok ilgili, {feature} hakkında detaylı sorular sordu.",
        "Demo'yu çok beğendi, hemen başlamak istiyor. {feature} konusunda heyecanlı.",
        "Görüşme harika geçti, {product} için kayıt formunu bekliyor.",
        "Lead son derece istekli, bir an önce {feature} görmek istiyor.",
        "Çok olumlu yaklaştı, ekip arkadaşlarını da getirmek istiyor.",
        "Müşteri 'tam aradığım şey buydu' dedi, sözleşmeyi konuşalım.",
        "Really excited about the {product}, asked how soon they can enroll.",
        "Great call! The lead loved the {feature} and wants to move forward today.",
        "Very engaged, asked detailed questions about the {feature} and pricing tiers.",
        "She said this is exactly what their team needs, ready to sign.",
        "Super keen, requested a demo and a proposal by end of week.",
        "Olumlu geçti, {feature} demosu sonrası karar verecek ama niyeti net.",
        "Hot lead - keeps replying within minutes and wants a follow-up call.",
        "Çok memnun kaldı, referans olarak başka firmalar önerdi.",
    ],
    "objection": [
        "Fiyatı yüksek buldu, indirim olup olmadığını sordu.",
        "Bütçesi bu çeyrek için uygun değil, {feature} pahalı geldi.",
        "Rakip firmanın teklifi daha ucuzmuş, ikna olması zor.",
        "Zamanlama kötü dedi, şimdi {product} için müsait değil.",
        "Yöneticisinin onayı gerekiyormuş, fiyat konusunda tereddütlü.",
        "Sözleşme şartlarına itiraz etti, esneklik istiyor.",
        "Too expensive for now, asked whether there is a cheaper {feature} tier.",
        "Has concerns about the price and wants to compare with a competitor.",
        "Said the timing is bad and the budget is already committed elsewhere.",
        "Pushed back on the contract length, wants a monthly option instead.",
        "Not convinced about the ROI, asked for a discount before deciding.",
        "ROI konusunda ikna olmadı, daha fazla kanıt istiyor.",
        "Bütçe onayı çıkmadı, fiyatı tekrar görüşmemiz gerekecek.",
        "Wants the {feature} but says the quote is over their budget.",
    ],
    "disengaged": [
        "Telefonu açmadı, üçüncü kez ulaşamadım.",
        "Kısa cevaplar veriyor, çok ilgisiz görünüyor.",
        "Şu an ilgilenmiyorum dedi, aramayın lütfen.",
        "Maile dönüş yapmadı, iki haftadır sessiz.",
        "Vazgeçti, başka bir çözüme geçmişler.",
        "Çok mesafeli, sadece 'bakıyorum' dedi ve kapattı.",
        "No response to the last three emails, seems to have gone cold.",
        "Said they're not interested anymore, please remove from the list.",
        "Very short replies, clearly not a priority for them right now.",
        "Asked us to stop contacting them, going with another vendor.",
        "Unsubscribed from emails and did not pick up the call.",
        "İlgilenmiyor, 'şimdilik gerek yok' dedi.",
        "Ghosted us after the first call, no replies since.",
        "Toplantıya katılmadı, takip mailine de dönmedi.",
    ],
    "neutral": [
        "Broşürü mail ile gönderdim, inceleyip dönecek.",
        "Görüşme planlandı, gelecek hafta salı saat 14:00.",
        "{product} hakkında genel bilgi istedi, standart sunumu paylaştım.",
        "İletişim bilgilerini güncelledik, takip notu bırakıldı.",
        "Müşteri bilgileri aldı, düşünüp haber verecek.",
        "Sent the {feature} brochure, awaiting their review.",
        "Scheduled a follow-up meeting for next week, no decision yet.",
        "Asked for general information about the {product}, shared the deck.",
        "Left a voicemail and sent a recap email, neutral so far.",
        "Standart bilgilendirme yapıldı, henüz net bir geri bildirim yok.",
        "Confirmed they received the proposal, will get back to us.",
        "Routine check-in, updated the CRM record with new contact details.",
    ],
}


@dataclass(frozen=True)
class InteractionDataset:
    """A labelled set of synthetic interaction notes for the sentiment model."""

    frame: pd.DataFrame  # columns: text, label, language

    @property
    def texts(self) -> list[str]:
        return self.frame["text"].tolist()

    @property
    def labels(self) -> list[str]:
        return self.frame["label"].tolist()


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(_PRODUCTS),
        feature=rng.choice(_FEATURES),
    )


def _detect_language(template: str) -> str:
    """Cheap heuristic: Turkish-specific characters / stopwords -> 'tr', else 'en'."""
    lowered = template.lower()
    if any(ch in lowered for ch in "çğıöşü") or any(
        w in lowered for w in (" dedi", "müşteri", "görüşme", "istiyor", "ilgili")
    ):
        return "tr"
    return "en"


def generate_interaction_dataset(
    n_per_class: int = 600,
    *,
    seed: int = RANDOM_SEED,
) -> InteractionDataset:
    """Generate a class-balanced set of synthetic interaction notes.

    Parameters
    ----------
    n_per_class:
        Number of notes to synthesise per intent class.
    seed:
        RNG seed for reproducibility.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label in SENTIMENT_LABELS:
        templates = _TEMPLATES[label]
        for _ in range(n_per_class):
            template = rng.choice(templates)
            rows.append(
                {
                    "text": _fill(template, rng),
                    "label": label,
                    "language": _detect_language(template),
                }
            )
    rng.shuffle(rows)
    return InteractionDataset(frame=pd.DataFrame(rows))


def sample_note(label: str, *, seed: int | None = None) -> str:
    """Return one synthetic note for a given intent label (used to build demo leads)."""
    rng = random.Random(seed)
    return _fill(rng.choice(_TEMPLATES[label]), rng)

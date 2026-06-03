"""Per-persona playbooks: a ready email draft + a call talk-track.

These are *templates* a rep can send/say in seconds, tailored to the lead category and
lightly personalised from the lead's features. Content is in Turkish (the CRM's working
language); placeholders are filled defensively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lead_priority.segmentation.rules import ApproachInsights

_DEFAULT_COURSE = "online eğitim programımız"


@dataclass(frozen=True)
class Playbook:
    """A concrete, ready-to-use outreach plan for one lead."""

    category: str
    channel: str
    email_subject: str
    email_body: str
    call_opening: str
    key_points: list[str]
    avoid: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "channel": self.channel,
            "email_subject": self.email_subject,
            "email_body": self.email_body,
            "call_opening": self.call_opening,
            "key_points": self.key_points,
            "avoid": self.avoid,
        }


def _topic(features: dict[str, Any]) -> str:
    spec = str(features.get("Specialization") or "").strip()
    return spec if spec else _DEFAULT_COURSE


# category -> builder producing (subject, body, call_opening, key_points)
def _ready_to_buy(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — başlamak için son adım",
        (
            "Merhaba,\n\n"
            f"{topic} konusundaki ilginiz için teşekkürler. Hızlıca başlayabilmeniz için "
            "kayıt adımlarını ve bu haftaki başlangıç tarihlerini paylaşıyorum. "
            "Uygun olduğunuz 15 dakikalık bir görüşmede tüm sorularınızı netleştirip "
            "kaydınızı tamamlayabiliriz.\n\nHangi saat sizin için uygun?"
        ),
        "Görüşmede çok olumluydunuz; bugün kaydı tamamlayalım diye aradım.",
        ["Net sonraki adım (kayıt/sözleşme)", "Bu haftaki başlangıç tarihi", "Hızlı kapanış"],
    )


def _price_objection(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — size uygun ödeme seçenekleri",
        (
            "Merhaba,\n\n"
            "Bütçe konusundaki geri bildiriminiz çok değerli. Programın sağladığı "
            "kazanımları ve geri dönüş (ROI) örneklerini, ayrıca taksitli ödeme ve "
            "kampanya seçeneklerini bir arada hazırladım. Yatırımın kendini nasıl "
            "amorti ettiğini birlikte değerlendirelim mi?"
        ),
        "Fiyatla ilgili çekinceniz vardı; size değer ve ödeme seçeneklerini netleştirmek için aradım.",
        ["Önce değer/ROI, sonra fiyat", "Taksit / ödeme planı", "Kampanya / indirim koşulları"],
    )


def _timing_objection(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — size uygun zamanda başlayın",
        (
            "Merhaba,\n\n"
            "Şu an yoğun olduğunuzu belirtmiştiniz. Esnek başlangıç tarihleri ve kendi "
            "hızınızda ilerleyebileceğiniz modüler yapı sayesinde programı yoğunluğunuza "
            "göre planlayabilirsiniz. Size uygun bir başlangıç tarihini birlikte belirleyelim mi?"
        ),
        "Zamanlama uygun değildi; esnek başlangıç seçeneklerini konuşmak için aradım.",
        ["Baskı yok, esnek başlangıç", "Modüler/kendi hızında ilerleme", "İleri tarihli hatırlatma"],
    )


def _competitor_objection(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — neden bizi tercih edenler memnun?",
        (
            "Merhaba,\n\n"
            "Alternatifleri değerlendirdiğinizi biliyorum. Bizi ayıran başlıca 3 noktayı "
            "(mentorluk, sertifika ve istihdam desteği) ve katılımcı başarı hikayelerini "
            "paylaşıyorum. Kıyaslamanızı netleştirecek kısa bir görüşme yapalım mı?"
        ),
        "Alternatifleri kıyaslıyordunuz; bizi ayıran noktaları net göstermek için aradım.",
        ["Fiyat savaşına girme", "2-3 net farklılaştırıcı", "Başarı hikayesi/referans"],
    )


def _information_seeker(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — incelediğiniz konulara özel kaynaklar",
        (
            "Merhaba,\n\n"
            "Sitemizde ilgilendiğiniz konulara dair müfredat detayını, örnek ders ve "
            "vaka çalışmalarını derledim. İhtiyacınıza en uygun içerikleri birlikte "
            "belirlemek için kısa bir görüşmeye ne dersiniz?"
        ),
        "İncelediğiniz konulara dair detayları ve uygun içerikleri paylaşmak için aradım.",
        ["Somut müfredat/örnek içerik", "İhtiyaca göre yönlendirme", "Düşük baskı, danışman tonu"],
    )


def _going_cold(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        "Hâlâ ilgileniyor musunuz? (kısa kontrol)",
        (
            "Merhaba,\n\n"
            f"{topic} konusunda bir süredir görüşemedik. Önceliğiniz değiştiyse tamamen "
            "anlarım. Yine de ilgileniyorsanız, kaldığınız yerden hızlıca ilerlemeniz için "
            "size özel bir başlangıç planı hazırlayabilirim. Tek bir 'evet' yeterli."
        ),
        "Bir süredir görüşemedik; hâlâ ilgileniyor musunuz diye kısa bir kontrol için yazdım.",
        ["Kişisel ve kısa", "Baskısız re-engagement", "Net ve kolay bir sonraki adım"],
    )


def _low_intent(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — dilediğinizde buradayız",
        (
            "Merhaba,\n\n"
            "Şu an için uygun olmayabilir; sizi sıkmak istemeyiz. İleride faydalı "
            "olabilecek ücretsiz kaynaklarımıza ve güncellemelere dilediğinizde göz "
            "atabilirsiniz. İhtiyaç olduğunda bir mesaj uzağınızdayız."
        ),
        "(Düşük öncelik) Aktif arama önerilmez; otomatik besleme (nurture) akışında tutun.",
        ["Düşük efor", "Otomatik içerik/nurture", "Zorlamadan kapıyı açık tut"],
    )


def _needs_nurturing(topic: str) -> tuple[str, str, str, list[str]]:
    return (
        f"{topic} — size faydalı olabilecek birkaç nokta",
        (
            "Merhaba,\n\n"
            f"{topic} hakkındaki kısa bir özet ve sık sorulan soruların yanıtlarını "
            "paylaşıyorum. Aklınızdaki soruları netleştirmek için kısa bir görüşmeye "
            "açıksanız, size uygun bir zaman ayarlayalım."
        ),
        "İhtiyacınızı netleştirip uygun seçenekleri sunmak için kısa bir görüşme önereceğim.",
        ["Keşif soruları", "Değer odaklı bilgilendirme", "Düşük baskılı sonraki adım"],
    )


_BUILDERS = {
    "ready_to_buy": _ready_to_buy,
    "price_objection": _price_objection,
    "timing_objection": _timing_objection,
    "competitor_objection": _competitor_objection,
    "information_seeker": _information_seeker,
    "going_cold": _going_cold,
    "low_intent": _low_intent,
    "needs_nurturing": _needs_nurturing,
}


def build_playbook(
    category: str,
    features: dict[str, Any],
    insights: ApproachInsights,
) -> Playbook:
    """Build a tailored playbook for a lead given its persona + approach insights."""
    builder = _BUILDERS.get(category, _needs_nurturing)
    topic = _topic(features)
    subject, body, call_opening, key_points = builder(topic)
    return Playbook(
        category=category,
        channel=insights.recommended_channel,
        email_subject=subject,
        email_body=body,
        call_opening=call_opening,
        key_points=key_points,
        avoid=insights.cautions,
    )

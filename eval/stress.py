"""Synthetic large PDFs with known 'needle' facts, similar-looking distractors, tables and repeated headers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

TOPICS = ["Procurement", "Security", "Facilities", "Onboarding", "Compliance", "Logistics", "Finance", "Support"]


@dataclass(frozen=True)
class Needle:
    page: int
    question: str
    answer: str  # substring that must appear in a correct answer
    text: str


def make_needles(pages: int) -> list[Needle]:
    """Three needles at the start, middle and end of the document."""
    spots = [min(3, pages), max(1, pages // 2), max(1, pages - 2)]
    facts = [
        ("Aurora", "7.5 percent"),
        ("Zephyr", "42 business days"),
        ("Cobalt", "USD 18,250"),
    ]
    kinds = [
        (
            "What is the late delivery penalty for Project {n}?",
            "Clause 47.3: the late delivery penalty for Project {n} is exactly {a} of the invoice value.",
        ),
        (
            "How long is the warranty claim window for Project {n}?",
            "Clause 52.1: the warranty claim window for Project {n} is exactly {a} from delivery.",
        ),
        (
            "What is the maximum liability cap for Project {n}?",
            "Clause 61.8: the maximum liability cap for Project {n} is exactly {a} in total.",
        ),
    ]
    answers = ["7.5 percent", "42 business days", "USD 18,250"]
    out = []
    for (name, _), (q, t), ans, page in zip(facts, kinds, answers, spots, strict=True):
        out.append(Needle(page, q.format(n=name), ans, t.format(n=name, a=ans)))
    return out


def _distractor(rng: random.Random, page: int) -> str:
    name = rng.choice(["Borealis", "Cirrus", "Delta", "Ember", "Falcon", "Granite", "Harbor"])
    kind = rng.choice(
        [
            f"The late delivery penalty for Project {name} is {rng.randint(1, 9)}.{rng.randint(0, 9)} percent of the invoice value.",
            f"The warranty claim window for Project {name} is {rng.randint(10, 90)} business days from delivery.",
            f"The maximum liability cap for Project {name} is USD {rng.randint(1, 90)},{rng.randint(100, 999)} in total.",
        ]
    )
    return kind


def build_stress_pdf(path: Path, pages: int, seed: int = 7) -> tuple[Path, list[Needle]]:
    rng = random.Random(seed)
    needles = make_needles(pages)
    by_page = {n.page: n for n in needles}
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    for p in range(1, pages + 1):
        c.setFont("Helvetica", 8)
        c.drawString(56, A4[1] - 36, "Contoso Master Services Agreement - Confidential")
        c.drawString(56, 30, f"Page {p} of {pages}")
        y = A4[1] - 80
        c.setFont("Helvetica-Bold", 15)
        c.drawString(56, y, f"Section {p}: {TOPICS[p % len(TOPICS)]} Terms")
        y -= 30
        c.setFont("Helvetica", 10.5)
        lines = [
            f"This section describes {TOPICS[p % len(TOPICS)].lower()} obligations for supplier group {rng.randint(100, 999)}.",
            _distractor(rng, p),
            f"Review meetings take place every {rng.randint(2, 12)} weeks unless both parties agree otherwise.",
            _distractor(rng, p),
            "All notices must be given in writing and delivered to the registered office.",
        ]
        if p in by_page:
            lines.insert(2, by_page[p].text)
        for ln in lines:
            c.drawString(56, y, ln[:105])
            y -= 20
            if len(ln) > 105:
                c.drawString(56, y, ln[105:210])
                y -= 20
            y -= 10
        c.showPage()
    c.save()
    return path, needles

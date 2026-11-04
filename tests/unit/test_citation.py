from app.services.citation import (
    build_citations,
    cited_numbers,
    is_not_available,
    strip_invalid_markers,
)
from app.services.prompt_builder import Passage


def P(n, doc="DOC001", name="a.pdf", page=1):
    return Passage(n, doc, name, page, "", f"text {n}", f"c{n}")


def test_cited_numbers_order_and_validity():
    assert cited_numbers("A [2] then B [1][2] and [9]", {1, 2, 3}) == [2, 1]
    assert cited_numbers("nothing here", {1}) == []


def test_strip_invalid_markers():
    assert strip_invalid_markers("Yes [1] and [7].", {1}) == "Yes [1] and ."


def test_citations_deduplicated_and_sorted():
    ps = [P(1, page=5), P(2, page=5), P(3, page=2), P(4, doc="DOC002", name="b.pdf", page=1)]
    cites, used = build_citations("x [1] y [2] z [3] w [4]", ps)
    assert [(c.document_id, c.page) for c in cites] == [("DOC001", 2), ("DOC001", 5), ("DOC002", 1)]
    assert len(used) == 4


def test_unused_chunks_not_listed():
    cites, used = build_citations("only [2]", [P(1, page=1), P(2, page=9)])
    assert [c.page for c in cites] == [9] and [p.number for p in used] == [2]


def test_fallback_to_top_passages_when_nothing_cited():
    cites, used = build_citations("no markers", [P(1, page=1), P(2, page=2), P(3, page=3)])
    assert [c.page for c in cites] == [1, 2]


def test_empty_passages():
    assert build_citations("answer", []) == ([], [])


def test_citations_come_from_metadata_not_model_text():
    cites, _ = build_citations("See page 99 of fake.pdf [1]", [P(1, name="real.pdf", page=4)])
    assert (cites[0].document, cites[0].page) == ("real.pdf", 4)


def test_not_available_detection():
    assert is_not_available("This information is not available in the uploaded documents.")
    assert is_not_available("The documents do not contain this.")
    assert not is_not_available("Annual leave is 24 days [1].")
    assert not is_not_available("x " * 200 + "does not contain")

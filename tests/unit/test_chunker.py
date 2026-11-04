import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.chunker import Chunker, SectionTracker, make_chunk_id, strip_section_prefix
from app.services.pdf_extractor import Block, PageContent


def page(n, *blocks):
    return PageContent(page_number=n, blocks=list(blocks))


def chunk(c, p, tracker=None):
    return c.chunk_page(p, document_id="DOC001", filename="a.pdf", tracker=tracker or SectionTracker())


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        Chunker(100, 100)


def test_chunk_size_and_metadata():
    c = Chunker(200, 30)
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk(c, page(4, Block(text)))
    assert len(chunks) > 3
    assert all(len(x.text) <= 200 for x in chunks)
    assert {x.page_number for x in chunks} == {4}
    assert all(x.document_id == "DOC001" and x.filename == "a.pdf" for x in chunks)
    assert [x.idx for x in chunks] == list(range(len(chunks)))


def test_overlap_between_consecutive_chunks():
    c = Chunker(120, 40)
    text = ". ".join(f"Sentence number {i} is here" for i in range(40))
    chunks = chunk(c, page(1, Block(text)))
    tail = chunks[0].text[-15:]
    assert any(tail.strip() in nxt.text for nxt in chunks[1:3])


def test_chunk_ids_deterministic():
    assert make_chunk_id("DOC007", 12, 3) == "DOC007-p0012-c003"
    c = Chunker(200, 20)
    p = page(2, Block("alpha beta gamma delta " * 30))
    assert [x.chunk_id for x in chunk(c, p)] == [x.chunk_id for x in chunk(c, p)]


def test_chunks_never_cross_pages():
    c = Chunker(200, 20)
    t = SectionTracker()
    a = chunk(c, page(1, Block("one " * 100)), t)
    b = chunk(c, page(2, Block("two " * 100)), t)
    assert all("two" not in x.text for x in a)
    assert all("one" not in x.text for x in b)


def test_very_short_pages_and_tiny_chunks_dropped():
    c = Chunker(200, 20)
    assert chunk(c, page(1, Block("3"))) == []
    assert chunk(c, page(1)) == []
    assert len(chunk(c, page(1, Block("A short but valid sentence.")))) == 1


def test_heading_sets_section_and_prefix():
    c = Chunker(300, 30)
    t = SectionTracker()
    p = page(
        1,
        Block("Leave Policy", "heading", 1),
        Block("Maternity Leave", "heading", 2),
        Block("Maternity leave lasts twenty six weeks for every eligible employee."),
    )
    chunks = chunk(c, p, t)
    assert chunks[0].section == "Leave Policy > Maternity Leave"
    assert chunks[0].stored_text.startswith("[Leave Policy > Maternity Leave]\n")
    assert strip_section_prefix(chunks[0].stored_text) == chunks[0].text


def test_section_path_carries_across_pages():
    c = Chunker(300, 30)
    t = SectionTracker()
    chunk(c, page(1, Block("Travel", "heading", 1)), t)
    nxt = chunk(c, page(2, Block("Hotels are limited to reasonable amounts per night.")), t)
    assert nxt[0].section == "Travel"


def test_sibling_heading_replaces_previous():
    t = SectionTracker()
    t.update(1, "A")
    t.update(2, "A1")
    t.update(2, "A2")
    assert t.path == "A > A2"
    t.update(1, "B")
    assert t.path == "B"


def test_sections_are_not_mixed_in_one_chunk():
    c = Chunker(700, 50)
    p = page(
        1,
        Block("Alpha", "heading", 1),
        Block("Alpha paragraph content is right here."),
        Block("Beta", "heading", 1),
        Block("Beta paragraph content is right here."),
    )
    chunks = chunk(c, p)
    assert len(chunks) == 2
    assert "Beta" not in chunks[0].text and chunks[1].section == "Beta"


def test_table_kept_as_own_chunk():
    c = Chunker(200, 20)
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    chunks = chunk(c, page(3, Block("Intro text that is long enough."), Block(table, "table")))
    assert [x.kind for x in chunks] == ["text", "table"]
    assert chunks[1].text == table


def test_table_chunks_keep_neighboring_rows_separate_with_header():
    c = Chunker(700, 50)
    table = "| type | paid days | notice |\n|---|---|---|\n| annual leave | 24 | 14 days |\n| study leave | 10 | 30 days |"

    chunks = chunk(c, page(2, Block(table, "table")))

    assert len(chunks) == 2
    assert all(chunk.text.startswith("| type | paid days | notice |\n|---|---|---|") for chunk in chunks)
    assert "annual leave" in chunks[0].text and "study leave" not in chunks[0].text
    assert "study leave" in chunks[1].text and "annual leave" not in chunks[1].text


def test_large_table_split_by_rows_with_header_repeated():
    c = Chunker(100, 10)
    rows = "\n".join(f"| row{i} | {i * 11} |" for i in range(80))
    table = "| name | value |\n|---|---|\n" + rows
    chunks = chunk(c, page(1, Block(table, "table")))
    assert len(chunks) == 80
    assert all(x.text.startswith("| name | value |\n|---|---|") for x in chunks)
    assert sum(x.text.count("| row") for x in chunks) == 80


def test_metadata_shape():
    c = Chunker(200, 20)
    ch = chunk(c, page(9, Block("Some reasonably long paragraph for the metadata test.")))[0]
    m = ch.metadata()
    assert m["document_id"] == "DOC001" and m["page_number"] == 9 and m["chunk_id"] == ch.chunk_id


@settings(max_examples=120, deadline=None)
@given(
    text=st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=0, max_size=1500),
    size=st.integers(60, 400),
)
def test_property_chunk_invariants(text, size):
    overlap = size // 6
    c = Chunker(size, overlap)
    chunks = chunk(c, page(5, Block(text)))
    for ch in chunks:
        assert ch.page_number == 5
        assert 0 < len(ch.text) <= size
        assert ch.text == ch.text.strip()
    assert len({ch.chunk_id for ch in chunks}) == len(chunks)

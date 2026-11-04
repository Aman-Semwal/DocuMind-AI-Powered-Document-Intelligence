from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.text_cleaner import clean_text


def test_collapses_whitespace_and_blank_lines():
    assert clean_text("Hello   world\n\n\n\nNext\tpara") == "Hello world\n\nNext para"


def test_empty_and_whitespace_only():
    assert clean_text("") == ""
    assert clean_text("   \n\t \n") == ""


def test_removes_control_and_zero_width_chars():
    assert clean_text("a\x00b\x07c​d") == "abcd"


def test_unicode_normalisation_ligatures():
    assert clean_text("oﬃce  policy") == "office policy"


def test_repairs_hyphenated_line_breaks():
    assert clean_text("inter-\nnational travel") == "international travel"


def test_keeps_real_hyphen_before_capital():
    assert clean_text("Anglo-\nSaxon") == "Anglo-Saxon"


def test_soft_wrapped_lines_are_joined():
    assert clean_text("This is a long\nsentence split\nacross lines.") == "This is a long sentence split across lines."


def test_list_items_stay_separate():
    out = clean_text("Rules:\n- first item\n- second item\n1. third")
    assert out.splitlines() == ["Rules:", "- first item", "- second item", "1. third"]


def test_header_footer_removed():
    bp = {"acme corp - policy", "page # of #"}
    text = "Acme Corp - Policy\nReal content here.\nPage 7 of 20"
    assert clean_text(text, bp) == "Real content here."


def test_footer_segment_after_wide_gap_removed():
    bp = {"page # of #", "confidential"}
    assert clean_text("Page 3 of 9        Confidential", bp) == ""
    assert clean_text("Body text   Page 3 of 9", bp) == "Body text"


def test_unicode_text_preserved():
    assert clean_text("Größe café 日本語 ₹500") == "Größe café 日本語 ₹500"


@settings(max_examples=150, deadline=None)
@given(st.text(max_size=400))
def test_property_output_invariants(text):
    out = clean_text(text)
    assert "\x00" not in out
    assert "\n\n\n" not in out
    assert out == out.strip()
    assert "\r" not in out
    assert "  " not in out.replace("\n", " ") or all("  " not in ln for ln in out.splitlines())


@settings(max_examples=100, deadline=None)
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=300))
def test_property_never_raises_and_returns_str(text):
    assert isinstance(clean_text(text, {"x"}), str)

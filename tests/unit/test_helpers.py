import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.api.ratelimit import RateLimiter
from app.config import Settings
from app.core.security import (
    looks_like_pdf,
    sanitize_filename,
    sanitize_question,
    stored_filename,
    strip_control_chars,
    verify_api_key,
)
from app.core.timing import StageTimer, percentile
from app.services.cache import LRUCache, SessionStore, normalize_question
from app.services.query_planner import decompose, is_summary_question
from app.services.textutil import estimate_tokens, stem, tokenize
from app.services.verifier import Verification, confidence_label, verify_answer
from run_all import wait_for_service


# ---------------------------------------------------------------- security
def test_api_key_comparison():
    assert verify_api_key("abc", ["x", "abc"])
    assert not verify_api_key("abd", ["abc"])
    assert not verify_api_key(None, ["abc"])
    assert not verify_api_key("", ["abc"])
    assert not verify_api_key("abc", [])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.pdf", "evil.pdf"),
        ("report final (1).pdf", "report final _1_.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("....", "document.pdf"),
        ("a\x00b.pdf", "ab.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_length_and_no_separators():
    out = sanitize_filename("x" * 500 + ".pdf")
    assert len(out) <= 120 and "/" not in out


def test_stored_filename_is_random_pdf():
    a, b = stored_filename(), stored_filename()
    assert a != b and a.endswith(".pdf") and "/" not in a


def test_sanitize_question_and_control_chars():
    assert sanitize_question("  hello \x00 \n world\t ") == "hello world"
    assert strip_control_chars("a\x07b") == "ab"


def test_magic_bytes():
    assert looks_like_pdf(b"%PDF-1.7 rest")
    assert looks_like_pdf(b"\n%PDF-1.4")
    assert not looks_like_pdf(b"MZ\x90\x00")
    assert not looks_like_pdf(b"")


# ---------------------------------------------------------------- config
def test_settings_lists_and_paths(tmp_path):
    s = Settings(_env_file=None, api_keys=" a , b ,,", cors_origins="http://x, http://y", data_dir=tmp_path)
    assert s.api_key_list == ["a", "b"]
    assert s.cors_origin_list == ["http://x", "http://y"]
    assert s.max_file_bytes == 100 * 1024 * 1024
    s.ensure_dirs()
    assert s.upload_dir.is_dir() and s.chroma_dir.is_dir() and s.preview_dir.is_dir()
    assert s.sqlite_path.parent == tmp_path and s.embed_cache_path.parent == tmp_path


def test_settings_validation():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FINAL_K", "7")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    s = Settings(_env_file=None)
    assert s.final_k == 7 and s.rerank_enabled is False


# ---------------------------------------------------------------- timing
def test_stage_timer():
    t = StageTimer()
    with t.stage("a"):
        time.sleep(0.01)
    t.add("a", 5)
    t.merge({"b": 2})
    d = t.as_dict()
    assert d["a"] >= 15 and d["b"] == 2 and d["total_ms"] >= d["a"] * 0.5


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([5], 50) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(101)), 95) == 95


def test_wait_for_service_rejects_health_from_another_process(monkeypatch):
    process = Mock()
    process.poll.return_value = 1
    monkeypatch.setattr("run_all.httpx.get", lambda *_args, **_kwargs: SimpleNamespace(status_code=200))
    assert not wait_for_service("http://localhost:8000/health", process)


def test_wait_for_service_accepts_healthy_child_process(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr("run_all.httpx.get", lambda *_args, **_kwargs: SimpleNamespace(status_code=200))
    assert wait_for_service("http://localhost:8000/health", process)


# ---------------------------------------------------------------- limiter
def test_rate_limiter_window():
    rl = RateLimiter(window_s=0.2)
    assert rl.allow("k", 2) and rl.allow("k", 2)
    assert not rl.allow("k", 2)
    assert rl.allow("other", 2)
    time.sleep(0.25)
    assert rl.allow("k", 2)
    assert rl.allow("k", 0)  # 0 disables the limit
    rl.reset()


# ---------------------------------------------------------------- cache
def test_lru_eviction_and_stats():
    c = LRUCache(2)
    c.put(("a",), 1)
    c.put(("b",), 2)
    assert c.get(("a",)) == 1
    c.put(("c",), 3)  # evicts b (least recently used)
    assert c.get(("b",)) is None and c.get(("a",)) == 1
    assert (c.hits, c.misses, len(c)) == (2, 1, 2)
    c.clear()
    assert len(c) == 0
    assert LRUCache(0).put(("x",), 1) is None


def test_cache_key_normalisation_and_versioning():
    k1 = LRUCache.make_key("How many days?", 1, 5, None, False)
    k2 = LRUCache.make_key("  how MANY days ", 1, 5, None, False)
    k3 = LRUCache.make_key("how many days", 2, 5, None, False)
    k4 = LRUCache.make_key("how many days", 1, 5, ["DOC002", "DOC001"], False)
    k5 = LRUCache.make_key("how many days", 1, 5, ["DOC001", "DOC002"], False)
    assert k1 == k2 and k1 != k3 and k1 != k4 and k4 == k5
    assert normalize_question("What's up?!") == "whats up"


def test_session_store():
    s = SessionStore(turns=2, max_sessions=2)
    assert s.history(None) == [] and s.history("x") == []
    s.add(None, "q", "a")
    s.add("a", "q1", "a1")
    s.add("a", "q2", "a2")
    s.add("a", "q3", "a3")
    assert s.history("a") == [("q2", "a2"), ("q3", "a3")]
    s.add("b", "q", "a")
    s.add("c", "q", "a")
    assert s.history("a") == []  # oldest session evicted
    s.clear("b")
    assert s.history("b") == []


# ---------------------------------------------------------------- planner
def test_summary_detection():
    for q in [
        "Summarize this document",
        "please summarise the policy",
        "What is this document about?",
        "What are the main topics?",
        "Give me an overview of the file",
    ]:
        assert is_summary_question(q), q
    assert not is_summary_question("How many sick days do I get?")


def test_decompose():
    assert decompose("How many sick days are there?") == []
    parts = decompose("How long is maternity leave? What is the hotel limit in London?")
    assert len(parts) == 2 and all(p.endswith("?") for p in parts)
    parts = decompose("What is the meal allowance and also how are mileage claims paid")
    assert len(parts) == 2
    assert len(decompose("What is a? What is b c d? What is e f g? What is h i j?", 3)) <= 3


# ---------------------------------------------------------------- textutil
def test_textutil():
    assert tokenize("Clause 4.2.1 e-mail, A/B") == ["clause", "4.2.1", "e-mail", "a/b"]
    assert stem("leaves") == stem("leave") == stem("leaving")
    assert stem("days") == "day" and stem("is") == "is" and stem("approved") == stem("approve")
    assert estimate_tokens("x" * 40) == 10 and estimate_tokens("") == 1


# ---------------------------------------------------------------- verifier
CTX = ["Annual leave is 24 days per year. Maternity leave is 26 weeks paid by Northwind Labs."]


def test_verifier_supported():
    v = verify_answer("Annual leave is 24 days per year [1].", CTX)
    assert v.ok and v.support_ratio == 1.0


def test_verifier_flags_wrong_number():
    v = verify_answer("Annual leave is 30 days per year.", CTX, question="Why does the policy give employees 30 days of annual leave?")
    assert not v.ok and v.unverified_numbers == ["30"]


def test_verifier_allows_a_question_number_repeated_as_a_condition():
    v = verify_answer(
        "A trip costing USD 12,000 needs CFO approval.",
        ["Trips costing more than USD 10,000 require Chief Financial Officer approval."],
        question="Who approves a trip costing USD 12,000?",
    )
    assert "12000" not in v.unverified_numbers


def test_verifier_flags_unsupported_sentence_and_entity():
    v = verify_answer("Employees receive free parking in Paris during winter holidays.", CTX)
    assert not v.ok and v.unsupported_sentences


def test_verifier_empty_answer():
    assert not verify_answer("   [1] ", CTX).ok


def test_confidence_label():
    ok = Verification()
    kw = {"high_threshold": 0.6, "low_threshold": 0.4}
    assert confidence_label(evidence=0.8, agreement=True, verification=ok, **kw) == "high"
    assert confidence_label(evidence=0.8, agreement=False, verification=ok, **kw) == "medium"
    assert confidence_label(evidence=0.3, agreement=True, verification=ok, **kw) == "low"
    bad = Verification(ok=False, unverified_numbers=["9"])
    assert confidence_label(evidence=0.9, agreement=True, verification=bad, **kw) == "low"
    soft = Verification(ok=False, unsupported_sentences=["x"], support_ratio=0.6)
    assert confidence_label(evidence=0.9, agreement=True, verification=soft, **kw) == "medium"
    assert confidence_label(evidence=0.5, agreement=False, verification=None, **kw) == "medium"

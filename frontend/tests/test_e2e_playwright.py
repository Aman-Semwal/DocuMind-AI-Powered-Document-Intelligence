"""Optional browser test against a LIVE UI:   python run_all.py --demo --seed   then   pytest -m browser"""

import os

import pytest

pytestmark = pytest.mark.browser
UI_URL = os.getenv("DOCQA_UI_URL", "http://localhost:8501").rstrip("/")


def test_upload_ask_and_see_citation():
    sync_api = pytest.importorskip("playwright.sync_api")
    import httpx

    try:
        httpx.get(f"{UI_URL}/_stcore/health", timeout=2)
    except httpx.HTTPError:
        pytest.skip(f"UI is not running at {UI_URL}")
    with sync_api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(UI_URL)
        page.wait_for_selector("textarea", timeout=30000)
        for question in (
            "How many days of annual leave do full-time employees get per year?",
            "How many weeks is maternity leave?",
            "What is the nightly hotel limit for Tier 1 cities such as London?",
        ):
            page.fill("textarea", question)
            page.keyboard.press("Enter")
            page.wait_for_timeout(4000)
        assert page.get_by_text("Northwind_").count() > 0  # a citation is visible
        browser.close()

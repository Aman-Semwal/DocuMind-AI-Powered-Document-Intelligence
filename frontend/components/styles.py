"""Responsive, local-font styling for the Streamlit workspace."""

import streamlit as st

CSS = """
<style>
:root {
  color-scheme: light;
  --paper: #f5f4f0;
  --surface: #fff;
  --surface-muted: #efeee8;
  --ink: #26312c;
  --ink-soft: #69746d;
  --line: #e2e0d8;
  --line-strong: #cfcdc3;
  --green: #28644f;
  --green-soft: #eaf2ec;
  --amber: #91651f;
  --amber-soft: #f5efdf;
  --red: #a34139;
  --red-soft: #f8eae7;
  --blue: #416a72;
  --blue-soft: #e9f0f0;
  --radius: 12px;
}

*, *::before, *::after { box-sizing: border-box; }
.stApp { background: var(--paper); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
[data-testid="stHeader"], [data-testid="stAppViewContainer"] { background: transparent !important; }
.block-container { max-width: 1160px; padding-top: 1.25rem; padding-bottom: 3rem; }
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] h5 { color: var(--ink); letter-spacing: -.025em; }
[data-testid="stCaptionContainer"] { color: var(--ink-soft); }

section[data-testid="stSidebar"] { background: var(--surface-muted); border-right: 1px solid var(--line); }
[data-testid="stSidebarContent"] { padding: 1.25rem 1rem 1.75rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stText"], [data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"], [data-testid="stSidebar"] .status-line,
[data-testid="stSidebar"] .brand-title, [data-testid="stSidebar"] .brand-tag,
[data-testid="stSidebar"] .muted { color: var(--ink) !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
[data-testid="stSidebar"] h5 { color: var(--green); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stSidebar"] hr { border-color: var(--line); }
.brand { display: flex; align-items: center; gap: .65rem; margin: .2rem 0 .45rem; }
.brand-logo { display: grid; place-items: center; width: 38px; height: 38px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); font-size: 1.15rem; }
.brand-title { color: var(--green) !important; font-size: 1.2rem; font-weight: 700; letter-spacing: -.04em; }
.brand-tag { color: var(--ink-soft) !important; font-size: .8rem; line-height: 1.5; margin: .35rem 0 1rem; }
.status-line { display: flex; align-items: center; font-size: .8rem; margin: .3rem 0; }
.dot { width: 8px; height: 8px; flex: 0 0 auto; margin-right: 8px; border-radius: 50%; background: #9ba39c; }
.dot.up { background: #34805f; }
.dot.down { background: #bd5349; }

.workspace-hero { position: relative; margin: .1rem 0 1.2rem; padding: clamp(1.4rem, 3vw, 2.1rem); border: 1px solid var(--line); border-radius: 18px; background: var(--surface); }
.workspace-kicker { display: flex; align-items: center; gap: .5rem; color: var(--green); font-size: .68rem; font-weight: 700; letter-spacing: .1em; }
.kicker-dot { width: 7px; height: 7px; border-radius: 50%; background: #4c896c; }
.workspace-hero h1 { margin: .65rem 0 .4rem; color: var(--ink); font-size: clamp(1.7rem, 3.3vw, 2.45rem); font-weight: 700; letter-spacing: -.05em; line-height: 1.12; }
.workspace-hero p { margin: 0; color: var(--ink-soft); font-size: .96rem; line-height: 1.55; }
.hero-private { position: absolute; right: 1.4rem; bottom: 1.4rem; padding: .38rem .7rem; border: 1px solid #dbe7dc; border-radius: 999px; background: #f3f7f2; color: var(--green); font-size: .72rem; white-space: nowrap; }

[data-testid="stTabs"] { margin-top: .25rem; }
[data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { min-height: 42px; padding: .45rem .85rem; color: var(--ink-soft); font-size: .86rem; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--green); font-weight: 650; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--green); }

[data-testid="stMetric"] { padding: .95rem 1rem; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); box-shadow: none; }
[data-testid="stMetricLabel"] { color: var(--ink-soft); font-size: .78rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 700; }
div[data-testid="stExpander"], div[data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
[data-testid="stChatInput"] { border: 1px solid var(--line-strong); border-radius: 15px; background: var(--surface); box-shadow: 0 4px 16px rgba(38,49,44,.05); }
[data-testid="stChatInput"]:focus-within { border-color: #81a28e; box-shadow: 0 0 0 3px rgba(40,100,79,.09); }
[data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input { color: var(--ink) !important; }
[data-testid="stChatInput"] textarea::placeholder, [data-testid="stTextInput"] input::placeholder { color: #89928c; }

.stButton > button, [data-testid="stDownloadButton"] > button,
[data-testid="stSidebar"] [data-baseweb="select"] > div { border: 1px solid var(--line-strong); border-radius: 9px; background: var(--surface); color: var(--ink); }
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover { border-color: #9ab0a0; background: #f5f8f4; color: var(--green); }
.stButton > button[kind="primary"] { border-color: var(--green); background: var(--green); color: #fff; }
.stButton > button[kind="primary"]:hover { border-color: #204e3e; background: #204e3e; color: #fff; }
[data-testid="stFileUploader"] section { border: 1px dashed #bacbbd; border-radius: var(--radius); background: #f9faf7; }
[data-testid="stFileUploader"] section:hover { border-color: #83a18d; background: #f3f7f2; }

.chip { display: inline-block; margin: 0 5px 5px 0; padding: 3px 9px; border: 1px solid var(--line); border-radius: 999px; background: #f7f7f4; color: var(--ink-soft); font-size: .7rem; }
.chip.ok, .badge.high { border-color: #d6e6d8; background: var(--green-soft); color: var(--green); }
.chip.warn, .badge.medium { border-color: #eadfbe; background: var(--amber-soft); color: var(--amber); }
.chip.bad, .badge.low { border-color: #efd7d3; background: var(--red-soft); color: var(--red); }
.chip.info { border-color: #d7e5e5; background: var(--blue-soft); color: var(--blue); }
.badge { display: inline-block; margin: 0 .4rem .4rem 0; padding: 4px 10px; border: 1px solid; border-radius: 999px; font-size: .72rem; font-weight: 650; }
.notfound { padding: .9rem 1rem; border: 1px solid #eddbd6; border-radius: var(--radius); background: #fbf2f0; color: var(--ink); line-height: 1.6; }
.notfound b { color: var(--red); }
.muted { color: var(--ink-soft); font-size: .78rem; }

[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"],
[role="listbox"], [role="listbox"] [role="option"] { background: #fff !important; color: var(--ink) !important; }

@media (max-width: 720px) {
  .block-container { padding: .75rem .7rem 2rem; }
  .workspace-hero { padding: 1.2rem; border-radius: 14px; }
  .workspace-hero p { max-width: 100%; font-size: .88rem; }
  .hero-private { position: static; display: inline-block; margin-top: .9rem; }
  [data-testid="stTabs"] button[role="tab"] { min-height: 40px; padding: .4rem .5rem; font-size: .78rem; }
  [data-testid="stSidebarContent"] { padding: 1rem .8rem 1.3rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

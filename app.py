"""
AuditTailor web interface (Streamlit).

    streamlit run app.py

UI only: paper upload → presentation style → pipeline (Phases 1–6) → download editable PPTX.
No methodology lives here. Local, key-free, deterministic research prototype.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import builtin_styles as BS  # noqa: E402
import pipeline as P  # noqa: E402

TITLE = "AuditTailor"
SUBTITLE = "Source-grounded scientific presentation generation"
PAPER_HELP = "PDF of the research paper to convert into a presentation."
STYLE_HELP = "Built-in reference profiles that define slide structure and presentation preferences."
CUSTOM_LABEL = "Use a custom reference presentation"
DECK_HELP = "Optional. A .pptx whose structure and style preferences should be followed instead of a built-in profile."
STYLE_DISCLAIMER = (
    "Built-in styles are bundled reference presentations used only to provide preference profiles; "
    "they are not scientifically validated."
)
MISSING_MSG = "Upload a research paper (PDF) to continue."
MISSING_CUSTOM_MSG = (
    "Upload a reference presentation, or disable “Use a custom reference presentation” "
    "to use a built-in style."
)
DEBUG = os.environ.get("AUDITTAILOR_DEBUG") == "1"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Professional, restrained palette
CSS = """
<style>
    /* Typography & base */
    html, body, [class*="css"]  {
        font-family: "Source Sans 3", "Source Sans Pro", "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        color: #1a1a1a !important;
        margin-bottom: 0.25rem !important;
        font-size: 1.75rem !important;
    }
    .at-subtitle {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.75rem;
        line-height: 1.45;
    }
    .at-section {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #444;
        margin: 1.5rem 0 0.65rem 0;
        border-bottom: 1px solid #e5e5e5;
        padding-bottom: 0.35rem;
    }
    .at-card {
        border: 1px solid #e2e2e2;
        border-radius: 6px;
        padding: 1rem 1.15rem;
        background: #fafafa;
        margin-bottom: 0.75rem;
    }
    .at-card p {
        margin: 0;
        color: #444;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .at-note {
        color: #666;
        font-size: 0.85rem;
        line-height: 1.4;
        margin: 0.35rem 0 0.75rem 0;
    }
    .at-result-header {
        border: 1px solid #d8e0d8;
        background: #f4f7f4;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        color: #1e3a1e;
        font-size: 0.95rem;
    }
    /* Metrics spacing */
    [data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #e8e8e8;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: #555;
    }
    /* Primary button */
    div.stButton > button[kind="primary"] {
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.01em;
        padding: 0.5rem 1rem;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #c8c8c8;
        border-radius: 6px;
        padding: 0.35rem;
        background: #fcfcfc;
    }
    /* Hide Streamlit chrome that reads as “AI chat” */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
"""


def pct(x: float) -> str:
    v = 100 * float(x)
    if abs(v - round(v)) < 0.05:
        return f"{round(v):.0f}%"
    return f"{v:.1f}%"


def advanced_settings() -> P.PipelineConfig:
    d = P.PipelineConfig()
    with st.expander("Advanced settings", expanded=False):
        st.caption("Optional. Defaults match the standard research configuration.")
        c1, c2 = st.columns(2)
        n_slides = c1.number_input("Content slides", 3, 40, d.n_slides, 1)
        rounds = c2.number_input("Maximum repair rounds", 0, 10, d.max_repair_rounds, 1)
        sup = c1.number_input("Minimum support ratio", 0.0, 1.0, float(d.min_support_ratio), 0.05)
        nf = c2.number_input("Minimum numeric fidelity", 0.0, 1.0, float(d.min_numeric_fidelity), 0.05)
        ov = c1.number_input("Maximum overflow rate", 0.0, 1.0, float(d.max_overflow_rate), 0.05)
    return P.PipelineConfig(
        n_slides=int(n_slides),
        max_repair_rounds=int(rounds),
        min_support_ratio=float(sup),
        min_numeric_fidelity=float(nf),
        max_overflow_rate=float(ov),
    )


def generate(paper, style_id: str, deck, cfg: P.PipelineConfig) -> None:
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    work = None
    bar = st.progress(0.0, text="Generating presentation…")
    with st.status("Running AuditTailor pipeline…", expanded=True) as status:
        try:
            staged = P.stage_inputs(
                paper.getvalue(),
                paper.name,
                style_id=style_id,
                custom_bytes=deck.getvalue() if deck is not None else None,
                custom_name=deck.name if deck is not None else None,
            )
            work = staged.work_dir
            total = len(P.STAGES)

            def progress(key: str, label: str) -> None:
                st.write(f"{label}")
                keys = [k for k, _ in P.STAGES]
                if key in keys:
                    bar.progress((keys.index(key) + 1) / total, text="Generating presentation…")

            res = P.run_pipeline(
                staged.paper_pdf,
                staged.reference_pptx,
                work / "out",
                cfg,
                progress,
                paper_name=paper.name,
                reference=staged.reference,
            )
            st.session_state["result"] = {
                "pptx": res.final_pptx.read_bytes(),
                "report": res.verification_json.read_bytes(),
                "history_json": res.repair_history_json.read_bytes(),
                "metrics": P.headline_metrics(res.verification, res.n_slides),
                "detail": P.detailed_metrics(res.verification),
                "repairs": P.summarize_repairs(res.history),
                "rounds": P.repair_round_rows(res.history),
                "reference": res.reference,
                "warnings": list(res.verification.get("warnings", [])),
                "stem": Path(paper.name).stem[:60] or "presentation",
            }
            status.update(label="Complete", state="complete", expanded=False)
        except Exception as e:
            st.session_state["error"] = {
                "message": P.describe_error(e),
                "trace": traceback.format_exc() if DEBUG else "",
            }
            status.update(label="Failed", state="error")
        finally:
            P.cleanup(work)
    bar.empty()


def show_error(err: dict) -> None:
    st.error(err["message"])
    if DEBUG and err.get("trace"):
        with st.expander("Technical details"):
            st.code(err["trace"], language="text")


def show_result(r: dict) -> None:
    st.markdown('<div class="at-result-header">Presentation generated successfully.</div>', unsafe_allow_html=True)
    if r.get("reference"):
        st.caption(f"Reference profile: {r['reference']['label']}")

    m = r["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Slides", m["Slides generated"])
    c2.metric("Source support", pct(m["Source Support"]))
    c3.metric("Numeric fidelity", pct(m["Numeric Fidelity"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Overflow rate", pct(m["Overflow Rate"]))
    c2.metric("Preference fidelity", pct(m["Preference Fidelity"]))
    c3.metric("Objective J(S)", f"{m['Final Objective Score']:.4f}", help="Lower is better")

    d = r["detail"]
    cols = st.columns(4)
    for col, name in zip(cols, ("Dalloc", "Dorder", "Dstyle", "Dpref")):
        col.metric(name, f"{d[name]:.4f}")

    st.markdown('<div class="at-section">Download</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download presentation (.pptx)",
            r["pptx"],
            file_name=f"{r['stem']}_AuditTailor.pptx",
            mime=PPTX_MIME,
            type="primary",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download verification report",
            r["report"],
            file_name=f"{r['stem']}_verification.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Repair details"):
        rp = r["repairs"]
        st.write(f"Repair rounds used: {rp['rounds_used']}")
        if rp["any_repairs"]:
            st.write("Repairs applied:")
            for a in rp["applied"]:
                st.write(f"· {a}")
            st.write(f"Objective J(S): {rp['initial_J']:.4f} → {rp['final_J']:.4f}")
        else:
            st.write("No repairs were necessary.")
        if r.get("rounds"):
            st.table(r["rounds"])
        st.caption(f"Stop reason: {rp['stop_reason']}")
        st.download_button(
            "Download repair history",
            r["history_json"],
            file_name=f"{r['stem']}_repair_history.json",
            mime="application/json",
        )
        if rp["remaining_failures"]:
            st.caption("Criteria not fully met: " + ", ".join(rp["remaining_failures"]))
        if r["warnings"]:
            st.caption("Verification notes: " + " · ".join(r["warnings"][:5]))


def choose_reference():
    st.markdown('<div class="at-section">Presentation style</div>', unsafe_allow_html=True)
    ids = [x.id for x in BS.STYLES]
    style_id = st.selectbox(
        "Style profile",
        ids,
        index=ids.index(BS.DEFAULT_STYLE_ID),
        format_func=lambda i: BS.get_style(i).label,
        key="style",
        help=STYLE_HELP,
    )
    style = BS.get_style(style_id)

    st.markdown(
        f'<div class="at-card"><p><strong>{style.label}</strong> — {style.description}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="at-note">Using the built-in {style.label.lower()} preference profile. '
        f'{STYLE_DISCLAIMER}</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Advanced: custom reference presentation", expanded=False):
        use_custom = st.checkbox(CUSTOM_LABEL, key="use_custom")
        deck = None
        if use_custom:
            st.caption(DECK_HELP)
            deck = st.file_uploader(
                "Reference presentation (.pptx)", type=["pptx"], help=DECK_HELP, key="custom_deck"
            )
            st.caption("When provided, this reference replaces the built-in style selected above.")
        else:
            deck = None

    use_custom = bool(st.session_state.get("use_custom", False))
    deck = st.session_state.get("custom_deck") if use_custom else None
    return style_id, use_custom, deck


def main() -> None:
    st.set_page_config(
        page_title=TITLE,
        page_icon=None,
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.title(TITLE)
    st.markdown(f'<p class="at-subtitle">{SUBTITLE}</p>', unsafe_allow_html=True)

    st.markdown('<div class="at-section">Research paper</div>', unsafe_allow_html=True)
    paper = st.file_uploader("Upload paper (PDF)", type=["pdf"], help=PAPER_HELP, key="paper")

    style_id, use_custom, deck = choose_reference()
    cfg = advanced_settings()

    problem = None
    if paper is None:
        problem = MISSING_MSG
    elif use_custom and deck is None:
        problem = MISSING_CUSTOM_MSG
    elif not use_custom and not BS.is_available(style_id):
        problem = (
            f'The built-in style “{BS.get_style(style_id).label}” is not installed. '
            "Choose another style or upload a custom reference presentation."
        )

    st.markdown('<div class="at-section">Generate</div>', unsafe_allow_html=True)
    if problem:
        st.info(problem)
    ready = problem is None
    if st.button("Generate presentation", type="primary", disabled=not ready, use_container_width=True):
        generate(paper, style_id, deck if use_custom else None, cfg)

    if "error" in st.session_state:
        show_error(st.session_state["error"])
    elif "result" in st.session_state:
        st.markdown('<div class="at-section">Results</div>', unsafe_allow_html=True)
        show_result(st.session_state["result"])


if __name__ == "__main__":
    main()

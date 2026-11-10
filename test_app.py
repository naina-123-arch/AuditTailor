"""
Web-interface tests (architecture level; no browser framework, no network).
Run:   python tests/test_app.py      (also collectable by pytest).

The synthetic paper / reference deck are copied to temp files with unrelated names to simulate uploads; the project's
sample files are never modified (hash-checked).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import pipeline as P                                  # noqa: E402
from suite_runner import run_suites, single_thread_env   # noqa: E402

PAPER = ROOT / "data" / "papers" / "synthetic_paper.pdf"
DECK = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_C = {}


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def run():
    """One upload-like pipeline run (files supplied by dynamic temp paths with unrelated names)."""
    if "r" not in _C:
        before = (sha(PAPER), sha(DECK))
        work = P.stage_uploads(PAPER.read_bytes(), "My Paper (final).PDF", DECK.read_bytes(), "my deck.pptx")
        events = []
        res = P.run_pipeline(work / "paper.pdf", work / "reference.pptx", work / "out", P.PipelineConfig(),
                             progress=lambda k, label: events.append((k, label)), paper_name=PAPER.name)
        _C["r"] = dict(work=work, res=res, events=events, before=before)
    return _C["r"]


# ------------------------------------------------------------------------------------- imports #
def test_app_and_pipeline_modules_import():
    import app                                          # noqa: F401  (importing must not run the UI)
    assert callable(app.main) and app.TITLE == "AuditTailor"
    assert app.SUBTITLE == "Source-grounded scientific presentation generation"
    assert app.CUSTOM_LABEL == "Use a custom reference presentation"
    assert "Upload a research paper" in app.MISSING_MSG
    assert "custom reference" in app.MISSING_CUSTOM_MSG.lower()
    import parser, section_tagger, qpp, planner, extractor, pptx_generator, verifier, repair, main   # noqa: E401,F401
    assert callable(P.run_pipeline)


def test_app_is_a_thin_layer_over_the_pipeline_module():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    for banned in ("import parser", "from parser", "from qpp", "from planner", "from extractor", "from verifier",
                   "from pptx_generator", "import repair", "from repair", "from main", "extract_qpp", "plan_slides",
                   "draft_deck", "generate_pptx", "repair_loop", "parse_pdf", "api_key", "openai", "anthropic"):
        assert banned not in src, banned
    assert "import pipeline as P" in src and "P.run_pipeline" in src and "P.stage_inputs" in src
    assert "type=[\"pdf\"]" in src and "type=[\"pptx\"]" in src and "st.download_button" in src


def test_ui_skeleton_without_uploads_disables_generate():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.value for t in at.title] == ["AuditTailor"]
    gen = next(b for b in at.button if "Generate" in b.label)
    assert gen.disabled
    infos = [i.value for i in at.info]
    assert any("research paper" in (i or "").lower() or "Upload" in (i or "") for i in infos) or any(
        "upload a research paper" in (i or "").lower() for i in infos
    )
    assert any("Advanced" in (e.label or "") for e in at.expander)


# ------------------------------------------------------------------------ uploads / temp files #
def test_uploads_are_validated_and_saved_under_safe_names():
    pdf, pptx = PAPER.read_bytes(), DECK.read_bytes()
    base = Path(tempfile.mkdtemp(prefix="at_app_base_"))
    work = P.stage_uploads(pdf, "../../etc/evil name.pdf", pptx, "..\\..\\x.pptx", base_dir=base)
    assert work.parent == base and sorted(p.name for p in work.iterdir()) == ["paper.pdf", "reference.pptx"]
    assert (work / "paper.pdf").read_bytes() == pdf and (work / "reference.pptx").read_bytes() == pptx
    assert not (base / "etc").exists() and not list(ROOT.glob("evil*"))
    w2 = P.stage_uploads(pdf, "a.pdf", pptx, "a.pptx", base_dir=base)
    assert w2 != work                                                       # every upload gets its own directory
    P.cleanup(work)
    P.cleanup(w2)
    assert not work.exists() and list(base.iterdir()) == []
    bad = [(b"not a pdf", "p.pdf", pptx, "d.pptx"), (pdf, "p.txt", pptx, "d.pptx"), (pdf, "p.pdf", pdf, "d.pptx"),
           (pdf, "p.pdf", pptx, "d.ppt"), (b"", "p.pdf", pptx, "d.pptx"), (pdf, "p.pdf", b"PK\x03\x04junk", "d.pptx"),
           (pdf, "p.pdf", pptx[:200], "d.pptx")]
    for args in bad:
        assert raises(ValueError, P.stage_uploads, *args, base), args[1::2]
    assert list(base.iterdir()) == []                                      # nothing left behind by rejected uploads
    shutil.rmtree(base)


# ------------------------------------------------------------------------------- pipeline run #
def test_pipeline_runs_on_dynamic_upload_like_paths_and_locates_outputs():
    r = run()
    res = r["res"]
    assert res.final_pptx == r["work"] / "out" / "phase6" / "final.pptx" and res.final_pptx.is_file()
    assert res.verification_json.is_file() and res.repair_history_json.is_file()
    for sub, f in (("phase1", "paper_parsed.json"), ("phase1", "reference_qpp.json"), ("phase2", "plan.json"),
                   ("phase3", "draft.json"), ("phase4", "audittailor_generated.pptx"), ("phase5", "verification.json")):
        assert (r["work"] / "out" / sub / f).is_file(), f
    from pptx import Presentation
    assert len(Presentation(str(res.final_pptx)).slides) == res.n_slides == 12        # opens; editable PPTX
    v = json.loads(res.verification_json.read_text(encoding="utf-8"))
    h = json.loads(res.repair_history_json.read_text(encoding="utf-8"))
    assert v["objective"] == res.verification["objective"] == h["final"]["J"] and h["schema"] == "audittailor.repair_history.v1"


def test_pipeline_matches_the_command_line_run():
    r = run()
    out = Path(tempfile.mkdtemp(prefix="at_app_cli_"))
    cmd = [sys.executable, str(ROOT / "src" / "main.py"), "--paper", str(PAPER), "--deck", str(DECK), "--out", str(out / "p1"),
           "--plan", "--n-slides", "10", "--plan-out", str(out / "plan.json"), "--draft", "--draft-out", str(out / "d.json"),
           "--pptx", "--pptx-out", str(out / "deck.pptx"), "--verify-out", str(out / "v.json"), "--repair",
           "--repair-out", str(out / "p6")]
    p = subprocess.run(cmd, capture_output=True, text=True, env=single_thread_env(), timeout=240)
    assert p.returncode == 0, p.stderr
    cli = json.loads((out / "p6" / "final_verification.json").read_text(encoding="utf-8"))
    ui = r["res"].verification
    for k in ("dalloc", "dorder", "dstyle", "dpref", "pf", "sr", "nf", "overflow_rate", "objective"):
        assert cli[k] == ui[k], k                                           # UI path == CLI path, metric for metric
    assert (out / "p6" / "final.pptx").read_bytes() == r["res"].final_pptx.read_bytes()
    shutil.rmtree(out)


def test_progress_stages_are_reported_in_order():
    r = run()
    assert [k for k, _ in r["events"]] == [k for k, _ in P.STAGES] == r["res"].stages_done
    assert [l for _, l in r["events"]] == ["Paper parsed", "Preferences extracted", "Slide plan created",
                                          "Source-grounded content drafted", "Presentation generated",
                                          "Presentation verified", "Repair completed"]


def test_sample_files_and_project_outputs_are_untouched():
    r = run()
    assert (sha(PAPER), sha(DECK)) == r["before"]
    before = {p: sha(p) for p in (ROOT / "outputs").rglob("*") if p.is_file()}
    run()                                                                   # (cached) no writes into the repo's outputs/
    assert {p: sha(p) for p in (ROOT / "outputs").rglob("*") if p.is_file()} == before
    assert not str(r["work"]).startswith(str(ROOT))                         # working dir is a system temp dir


# ---------------------------------------------------------------------- config / errors / display #
def test_config_defaults_come_from_existing_modules_and_flow_through():
    import repair as R
    from main import DEFAULT_LAMBDA
    c = P.PipelineConfig()
    assert (c.max_repair_rounds, c.min_support_ratio, c.min_numeric_fidelity, c.max_overflow_rate, c.lam) == \
           (R.MAX_REPAIR_ROUNDS, R.MIN_SUPPORT_RATIO, R.MIN_NUMERIC_FIDELITY, R.MAX_OVERFLOW_RATE, DEFAULT_LAMBDA)
    work = P.stage_uploads(PAPER.read_bytes(), "p.pdf", DECK.read_bytes(), "d.pptx")
    res = P.run_pipeline(work / "paper.pdf", work / "reference.pptx", work / "out",
                         P.PipelineConfig(max_repair_rounds=1, n_slides=8))
    assert res.history["config"]["max_rounds"] == 1
    assert res.verification["details"]["plan"]["slide_allocation"] and \
        sum(res.verification["details"]["plan"]["slide_allocation"].values()) == 8
    P.cleanup(work)


def test_display_name_is_sanitised():
    assert P.display_name("../../etc/pass wd<script>.pdf") == "pass wd_script_.pdf"
    assert P.display_name("") == "paper.pdf" and len(P.display_name("a" * 500 + ".pdf")) <= 80


def test_failures_are_reported_as_friendly_pipeline_errors():
    work = Path(tempfile.mkdtemp(prefix="at_app_err_"))
    (work / "paper.pdf").write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    shutil.copy(DECK, work / "reference.pptx")
    try:
        P.run_pipeline(work / "paper.pdf", work / "reference.pptx", work / "out")
        assert False, "expected PipelineError"
    except P.PipelineError as e:
        assert e.stage == "parse" and "Something went wrong while reading the research paper" in str(e)
        assert e.__cause__ is not None                                      # original exception kept for 'Technical details'
    assert raises(P.PipelineError, P.run_pipeline, work / "missing.pdf", work / "reference.pptx", work / "o2")
    shutil.rmtree(work)


def test_displayed_values_come_from_the_verification_and_history_files():
    r = run()["res"]
    v = json.loads(r.verification_json.read_text(encoding="utf-8"))
    m = P.headline_metrics(v, r.n_slides)
    assert m == {"Slides generated": 12, "Source Support": v["sr"], "Numeric Fidelity": v["nf"],
                 "Overflow Rate": v["overflow_rate"], "Preference Fidelity": v["pf"], "Final Objective Score": v["objective"]}
    s = P.summarize_repairs(json.loads(r.repair_history_json.read_text(encoding="utf-8")))
    assert s["rounds_used"] == 0 and s["applied"] == [] and not s["any_repairs"]              # clean deck: no repairs
    h = {"rounds_used": 2, "stop_reason": "x", "initial": {"J": 0.5}, "final": {
        "J": 0.1, "improved_over_initial": True, "repair_types_applied": ["overflow", "support_fidelity"], "remaining_failures": []}}
    s2 = P.summarize_repairs(h)
    assert s2["applied"] == ["Overflow adjustment", "Source-support / numeric correction"] and s2["any_repairs"]


# --------------------------------------------------------------------------------- earlier phases #
def test_earlier_phase_suites_still_pass():
    run_suites((1, 2, 3, 4, 5, 6), timeout=600)        # each suite in its own subprocess, concurrently, with a timeout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import time
    for t in tests:
        t0 = time.time()
        t()
        print(f"  ok  {t.__name__}  ({time.time() - t0:.1f}s)")
    for d in [_C.get("r", {}).get("work")]:
        P.cleanup(d)
    print(f"ALL APP TESTS PASSED ({len(tests)} tests)")

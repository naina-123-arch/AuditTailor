"""
Tests for the optional-reference workflow (paper PDF + built-in presentation style, or the user's own reference deck).
Run:   python tests/test_ui_reference.py      (also collectable by pytest).

Covers: PDF upload, default Academic Research style, other built-in styles, custom reference upload, custom overriding
built-in, missing built-in file, invalid PDF / PPTX, end-to-end runs (built-in and custom reference), the separation of
reference (preferences) and template (look), and the Streamlit page itself (Streamlit's AppTest, simulated uploads).
No network, no LLM.  The project's sample / built-in files are hash-checked and never modified.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import builtin_styles as BS                          # noqa: E402
import pipeline as P                                 # noqa: E402

PAPER = ROOT / "data" / "papers" / "synthetic_paper.pdf"
DECK = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_C: dict = {}


class SkipTest(Exception):
    pass


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    return None


def pipeline_run(key: str, **stage_kw):
    """One cached end-to-end run (stage_inputs -> run_pipeline) per scenario."""
    if key not in _C:
        staged = P.stage_inputs(PAPER.read_bytes(), "My Paper.pdf", **stage_kw)
        res = P.run_pipeline(staged.paper_pdf, staged.reference_pptx, staged.work_dir / "out", P.PipelineConfig(),
                             paper_name="My Paper.pdf", reference=staged.reference)
        _C[key] = (staged, res)
    return _C[key]


def new_app(timeout=300):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout)
    return at


def upload_ok(at) -> bool:
    return hasattr(at, "file_uploader")


# ------------------------------------------------------------------------- bundled styles #
def test_builtin_styles_are_bundled_and_valid():
    assert len(BS.STYLES) == 5 and BS.DEFAULT_STYLE_ID == "academic_research"
    assert [s.label for s in BS.STYLES] == ["Academic Research", "Technical / AI", "Conference", "Minimal Professional",
                                            "Detailed Research"]
    assert len(BS.available_styles()) == 5 and BS.BUILTIN_DIR == ROOT / "data" / "reference_decks" / "builtin"
    from qpp import extract_qpp
    from section_tagger import CANONICAL_SECTIONS
    from pptx import Presentation
    prof = {}
    for s in BS.STYLES:
        path = BS.builtin_path(s.id)
        assert path.is_file() and path.suffix == ".pptx" and s.description
        q = extract_qpp(path).to_dict()
        w = q["theta"]["w"]
        assert abs(sum(w.values()) - 1) < 1e-4 and set(w) == set(CANONICAL_SECTIONS) and all(v > 0 for v in w.values())
        assert q["theta"]["pi"] == list(CANONICAL_SECTIONS)
        names = {l.name for l in Presentation(str(path)).slide_layouts}      # usable as the Phase 4 / 6 template too
        assert {"Title Slide", "Title and Content"} <= names
        prof[s.id] = q
    # the styles really are different profiles (structure only; no claim that they are 'validated')
    bb = {k: v["mean_bullets"] for k, v in prof.items()}
    assert bb["minimal_professional"] < bb["conference"] < bb["academic_research"] < bb["detailed_research"]
    assert prof["technical_ai"]["theta"]["w"]["Method"] > prof["academic_research"]["theta"]["w"]["Method"]
    assert prof["conference"]["theta"]["w"]["Results"] > prof["academic_research"]["theta"]["w"]["Results"]
    assert raises(ValueError, BS.get_style, "nope")


# ----------------------------------------------------------------------------- staging #
def test_pdf_upload_default_style_is_academic_research_builtin():
    before = sha(BS.builtin_path("academic_research"))
    st = P.stage_inputs(PAPER.read_bytes(), "../weird name.PDF")               # no style given, no custom deck
    try:
        assert sorted(p.name for p in st.work_dir.iterdir()) == ["paper.pdf", "reference.pptx"]
        assert st.paper_pdf.read_bytes() == PAPER.read_bytes()
        assert st.reference.kind == "builtin" and st.reference.style_id == "academic_research"
        assert st.reference.label == "Academic Research (built-in)"
        assert st.reference_pptx.read_bytes() == BS.builtin_path("academic_research").read_bytes()
    finally:
        P.cleanup(st.work_dir)
    assert sha(BS.builtin_path("academic_research")) == before                # the bundled file is only copied


def test_user_can_select_another_builtin_style():
    for s in BS.STYLES:
        st = P.stage_inputs(PAPER.read_bytes(), "p.pdf", style_id=s.id)
        try:
            assert st.reference.label == f"{s.label} (built-in)" and st.reference.style_id == s.id
            assert st.reference_pptx.read_bytes() == BS.builtin_path(s.id).read_bytes()
        finally:
            P.cleanup(st.work_dir)
    assert raises(ValueError, P.stage_inputs, PAPER.read_bytes(), "p.pdf", style_id="not_a_style")


def test_custom_reference_upload_works_and_overrides_the_builtin_style():
    st = P.stage_inputs(PAPER.read_bytes(), "p.pdf", style_id="technical_ai", custom_bytes=DECK.read_bytes(),
                        custom_name="my deck.pptx")
    try:
        assert st.reference.kind == "custom" and st.reference.label == "Custom uploaded presentation"
        assert st.reference.style_id is None
        assert st.reference_pptx.read_bytes() == DECK.read_bytes()            # custom wins over the selected built-in
        assert st.reference_pptx.read_bytes() != BS.builtin_path("technical_ai").read_bytes()
    finally:
        P.cleanup(st.work_dir)


def test_missing_builtin_reference_is_handled_gracefully():
    empty = Path(tempfile.mkdtemp(prefix="at_nobuiltin_"))
    base = Path(tempfile.mkdtemp(prefix="at_base_"))
    try:
        assert BS.available_styles(empty) == [] and not BS.is_available("academic_research", empty)
        e = raises(ValueError, P.stage_inputs, PAPER.read_bytes(), "p.pdf", base_dir=base, builtin_dir=empty)
        assert e is not None and "not installed" in str(e) and "Academic Research" in str(e)
        assert list(base.iterdir()) == []                                      # nothing left behind
        # a custom reference still works when the built-in folder is empty
        st = P.stage_inputs(PAPER.read_bytes(), "p.pdf", custom_bytes=DECK.read_bytes(), custom_name="d.pptx",
                            base_dir=base, builtin_dir=empty)
        P.cleanup(st.work_dir)
        assert P.describe_error(e) == str(e)
    finally:
        shutil.rmtree(empty, ignore_errors=True)
        shutil.rmtree(base, ignore_errors=True)


def test_invalid_pdf_is_rejected():
    import pymupdf
    base = Path(tempfile.mkdtemp(prefix="at_base_"))
    blank = pymupdf.open()
    blank.new_page()
    blank_bytes = blank.tobytes()                                              # a real PDF without any text
    bad = [(b"just some text", "p.pdf", "does not look like a valid PDF"),
           (PAPER.read_bytes(), "p.txt", "must be a PDF"), (b"", "p.pdf", "is empty"),
           (DECK.read_bytes(), "p.pdf", "does not look like a valid PDF"),
           (b"%PDF-1.4\nnot really a pdf at all", "p.pdf", "could not be opened"),
           (blank_bytes, "p.pdf", "No selectable text")]
    try:
        for data, name, expect in bad:
            e = raises(ValueError, P.stage_inputs, data, name, base_dir=base)
            assert e is not None and expect in str(e), (name, expect, e)
            assert "Traceback" not in str(e)
        assert list(base.iterdir()) == []
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_invalid_pptx_reference_is_rejected():
    import io
    import zipfile
    from pptx import Presentation
    base = Path(tempfile.mkdtemp(prefix="at_base_"))
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("hello.txt", "hi")
    title_only = io.BytesIO()                                                  # valid PPTX, but no content slide to learn from
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[0]).shapes.title.text = "Only a title"
    prs.save(title_only)
    bad = [(PAPER.read_bytes(), "d.pptx", "does not look like a valid PPTX"), (b"hello", "d.pptx", "does not look like"),
           (DECK.read_bytes(), "d.ppt", "must be a PPTX"), (b"", "d.pptx", "is empty"),
           (DECK.read_bytes()[:300], "d.pptx", "not a valid PPTX"), (zbuf.getvalue(), "d.pptx", "not a valid PPTX"),
           (title_only.getvalue(), "d.pptx", "could not be analysed")]
    try:
        for data, name, expect in bad:
            e = raises(ValueError, P.stage_inputs, PAPER.read_bytes(), "p.pdf", custom_bytes=data, custom_name=name,
                       base_dir=base)
            assert e is not None and expect in str(e), (name, expect, e)
        assert list(base.iterdir()) == []
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ------------------------------------------------------------------------ end-to-end runs #
def _profile(q: dict) -> dict:
    """The preference profile itself (theta + mean bullets); the QPP file also records the deck's file name."""
    return {"theta": q["theta"], "mean_bullets": q["mean_bullets"]}


def _check_run(staged, res, label):
    from pptx import Presentation
    assert res.final_pptx.is_file() and len(Presentation(str(res.final_pptx)).slides) == res.n_slides > 3   # opens; editable
    assert res.verification_json.name == "final_verification.json" and res.verification_json.is_file()
    assert res.repair_history_json.name == "repair_history.json" and res.repair_history_json.is_file()
    ver = json.loads(res.verification_json.read_text(encoding="utf-8"))
    hist = json.loads(res.repair_history_json.read_text(encoding="utf-8"))
    d = P.detailed_metrics(ver)                                                # the displayed numbers = the file's numbers
    assert list(d) == ["Dalloc", "Dorder", "Dstyle", "Dpref", "PF", "SR", "NF", "O", "J(S)"]
    assert d["Dalloc"] == ver["dalloc"] and d["J(S)"] == ver["objective"] == hist["final"]["J"]
    assert abs(d["PF"] - (1 - d["Dpref"])) < 1e-5 and 0 <= d["SR"] <= 1 and 0 <= d["NF"] <= 1
    assert P.repair_round_rows(hist) == [] if hist["rounds_used"] == 0 else len(P.repair_round_rows(hist)) == hist["rounds_used"]
    v5 = json.loads((res.output_dir / "phase5" / "verification.json").read_text(encoding="utf-8"))
    assert res.reference["label"] == label and v5["inputs"]["reference_qpp"] == label     # recorded for reproducibility
    assert hist["inputs"]["reference_qpp"] == label
    qpp = json.loads((res.output_dir / "phase1" / "reference_qpp.json").read_text(encoding="utf-8"))
    return qpp, ver


def test_end_to_end_with_builtin_reference():
    staged, res = pipeline_run("builtin")
    qpp, ver = _check_run(staged, res, "Academic Research (built-in)")
    from qpp import extract_qpp
    assert _profile(qpp) == _profile(extract_qpp(BS.builtin_path("academic_research")).to_dict())   # QPP came from the built-in deck
    w = qpp["theta"]["w"]
    plan = json.loads((res.output_dir / "phase2" / "plan.json").read_text(encoding="utf-8"))
    assert sum(plan["slide_allocation"].values()) == 10 and w["Method"] > 0


def test_end_to_end_with_custom_reference():
    staged, res = pipeline_run("custom", custom_bytes=DECK.read_bytes(), custom_name="mine.pptx")
    qpp, ver = _check_run(staged, res, "Custom uploaded presentation")
    from qpp import extract_qpp
    assert _profile(qpp) == _profile(extract_qpp(DECK).to_dict())              # QPP came from the uploaded deck
    assert _profile(qpp) != _profile(json.loads((pipeline_run("builtin")[1].output_dir / "phase1" / "reference_qpp.json")
                                                .read_text(encoding="utf-8")))   # ... not from the built-in one


def test_different_styles_give_different_plans_through_the_same_pipeline():
    a = pipeline_run("builtin")[1]
    t = pipeline_run("tech", style_id="technical_ai")[1]
    pa = json.loads((a.output_dir / "phase2" / "plan.json").read_text(encoding="utf-8"))["slide_allocation"]
    pt = json.loads((t.output_dir / "phase2" / "plan.json").read_text(encoding="utf-8"))["slide_allocation"]
    assert pa != pt and pt["Method"] >= pa["Method"]


def test_reference_and_template_are_separate_roles():
    """Reference deck -> preferences (QPP); template -> look of the generated PPTX.  Same pipeline, two files."""
    from pptx import Presentation
    tmp = Path(tempfile.mkdtemp(prefix="at_tmpl_"))
    try:
        wide = Presentation(str(DECK))                                         # a visual template with a different slide size
        wide.slide_width = 12192000
        wide.save(str(tmp / "wide.pptx"))
        staged = P.stage_inputs(PAPER.read_bytes(), "p.pdf", style_id="conference")
        res = P.run_pipeline(staged.paper_pdf, staged.reference_pptx, staged.work_dir / "out", P.PipelineConfig(),
                             reference=staged.reference, template_pptx=tmp / "wide.pptx")
        out = Presentation(str(res.final_pptx))
        assert out.slide_width == 12192000 != Presentation(str(staged.reference_pptx)).slide_width     # look from the template
        from qpp import extract_qpp
        qpp = json.loads((res.output_dir / "phase1" / "reference_qpp.json").read_text(encoding="utf-8"))
        assert _profile(qpp) == _profile(extract_qpp(BS.builtin_path("conference")).to_dict())   # preferences from the reference
        assert raises(P.PipelineError, P.run_pipeline, staged.paper_pdf, staged.reference_pptx, tmp / "o",
                      template_pptx=tmp / "missing.pptx")
        P.cleanup(staged.work_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bundled_and_sample_files_untouched_by_runs():
    pipeline_run("builtin")
    before = {p: sha(p) for p in (ROOT / "data").rglob("*") if p.is_file()}
    pipeline_run("custom", custom_bytes=DECK.read_bytes(), custom_name="mine.pptx")
    assert {p: sha(p) for p in (ROOT / "data").rglob("*") if p.is_file()} == before


def test_friendly_errors_hide_tracebacks_and_class_names():
    work = Path(tempfile.mkdtemp(prefix="at_err_"))
    (work / "paper.pdf").write_bytes(b"%PDF-1.4\nbroken")
    shutil.copy(DECK, work / "reference.pptx")
    try:
        e = raises(P.PipelineError, P.run_pipeline, work / "paper.pdf", work / "reference.pptx", work / "out")
        msg = P.describe_error(e)
        assert msg.startswith("We could not read the research paper.") and "Traceback" not in msg and "Error:" not in msg
        assert "went wrong" in P.describe_error(RuntimeError("secret internals")) and "secret" not in \
               P.describe_error(RuntimeError("secret internals"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_repair_history_rows_and_rendering():
    """A history with a real-looking repair round (schema of repair_history.json) is tabulated and rendered."""
    hist = {"rounds_used": 2, "stop_reason": "all_criteria_satisfied", "initial": {"J": 0.9},
            "final": {"J": 0.3, "improved_over_initial": True, "repair_types_applied": ["overflow"], "remaining_failures": []},
            "rounds": [
                {"round": 1, "families_attempted": ["overflow", "support"], "previous_J": 0.9, "new_J": 0.5, "improved": True,
                 "actions": [{"changed": True}, {"changed": False}]},
                {"round": 2, "families_attempted": ["support"], "previous_J": 0.5, "new_J": None, "improved": False,
                 "actions": [{"changed": False}], "note": "no repair changed the candidate"}]}
    rows = P.repair_round_rows(hist)
    assert [r["Round"] for r in rows] == [1, 2] and rows[0]["Repair types tried"] == "overflow, support"
    assert rows[0]["Actions that changed the deck"] == 1 and rows[0]["J before"] == 0.9 and rows[0]["J after"] == 0.5
    assert rows[1]["Kept (J decreased)"] is False and rows[1]["Note"].startswith("no repair")
    assert P.repair_round_rows({"rounds": []}) == []
    from streamlit.testing.v1 import AppTest

    ns = {"ROOT_STR": str(ROOT), "FAKE_REF": {"label": "Academic Research (built-in)"}, "ROWS": rows,
          "DETAIL": {"Dalloc": 0.0, "Dorder": 0.0, "Dstyle": 0.2, "Dpref": 0.08, "PF": 0.92, "SR": 1.0, "NF": 1.0, "O": 0.0, "J(S)": 0.3},
          "SUMMARY": P.summarize_repairs(hist)}
    src = "import streamlit as st\n" + "\n".join(f"{k} = {v!r}" for k, v in ns.items()) + "\n" + \
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(ROOT_STR) / 'src'))\nsys.path.insert(0, ROOT_STR)\n" \
        "import app\napp.show_result({'pptx': b'x', 'report': b'{}', 'history_json': b'{}', 'reference': FAKE_REF, " \
        "'metrics': {'Slides generated': 12, 'Source Support': 1.0, 'Numeric Fidelity': 1.0, 'Overflow Rate': 0.0, " \
        "'Preference Fidelity': 0.92, 'Final Objective Score': 0.3}, 'detail': DETAIL, 'repairs': SUMMARY, 'rounds': ROWS, " \
        "'warnings': [], 'stem': 'paper'})\n"
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Reference profile:" in (c.value or "") and "Academic" in (c.value or "") for c in at.caption)
    assert {"Dalloc", "Dorder", "Dstyle", "Dpref"} <= {m.label for m in at.metric} and len(at.table) == 1


# --------------------------------------------------------------------------- Streamlit page #
def test_page_defaults_and_style_selection():
    at = new_app(60).run()
    assert not at.exception
    sb = at.selectbox(key="style")
    assert sb.value == "academic_research" and sb.options == [s.label for s in BS.STYLES] or \
        [str(o) for o in sb.options] == [s.label for s in BS.STYLES]
    assert not at.checkbox(key="use_custom").value
    if upload_ok(at):
        assert len(at.file_uploader) == 1 and at.file_uploader[0].key == "paper"     # only the PDF is asked for
    gen = next(b for b in at.button if b.label == "Generate presentation")
    assert gen.disabled and "Upload a research paper (PDF) to continue." in [i.value for i in at.info]
    # Style card / disclaimer rendered as markdown or caption
    blob = " ".join([(getattr(c, "value", None) or "") for c in list(at.caption) + list(getattr(at, "markdown", []) or [])])
    assert "not scientifically validated" in blob or any("not scientifically validated" in (c.value or "") for c in at.caption)
    at.selectbox(key="style").select("technical_ai").run()                     # choose another built-in style
    assert at.selectbox(key="style").value == "technical_ai"


def test_page_custom_reference_toggle():
    at = new_app(60).run()
    if not upload_ok(at):
        raise SkipTest("installed Streamlit has no AppTest file_uploader support")
    at.checkbox(key="use_custom").check().run()
    assert [u.key for u in at.file_uploader] == ["paper", "custom_deck"]
    at.file_uploader(key="paper").upload("paper.pdf", PAPER.read_bytes(), "application/pdf").run()
    gen = next(b for b in at.button if b.label == "Generate presentation")
    assert gen.disabled and any("reference presentation" in (i.value or "").lower() for i in at.info)
    at.file_uploader(key="custom_deck").upload("d.pptx", DECK.read_bytes()).run()
    assert not next(b for b in at.button if b.label == "Generate presentation").disabled
    at.checkbox(key="use_custom").uncheck().run()                              # back to the built-in style
    assert [u.key for u in at.file_uploader] == ["paper"]


def test_page_missing_builtin_style_disables_generate_with_friendly_message():
    if not upload_ok(new_app(30).run()):
        raise SkipTest("installed Streamlit has no AppTest file_uploader support")
    empty = Path(tempfile.mkdtemp(prefix="at_nobuiltin_"))
    saved = BS.BUILTIN_DIR
    try:
        BS.BUILTIN_DIR = empty
        at = new_app(60).run()
        at.file_uploader(key="paper").upload("paper.pdf", PAPER.read_bytes()).run()
        gen = next(b for b in at.button if b.label == "Generate presentation")
        assert gen.disabled and any("is not installed" in i.value for i in at.info) and not at.exception
    finally:
        BS.BUILTIN_DIR = saved
        shutil.rmtree(empty, ignore_errors=True)


def test_page_invalid_uploads_show_friendly_errors():
    if not upload_ok(new_app(30).run()):
        raise SkipTest("installed Streamlit has no AppTest file_uploader support")
    at = new_app(120).run()
    at.file_uploader(key="paper").upload("fake.pdf", b"this is not a pdf").run()
    next(b for b in at.button if b.label == "Generate presentation").click().run()
    assert not at.exception and at.error and "does not look like a valid PDF" in at.error[0].value
    assert "Traceback" not in at.error[0].value and not at.expander or all("Technical" not in e.label for e in at.expander)
    at = new_app(120).run()
    at.checkbox(key="use_custom").check().run()
    at.file_uploader(key="paper").upload("paper.pdf", PAPER.read_bytes()).run()
    at.file_uploader(key="custom_deck").upload("deck.pptx", b"PK\x03\x04 definitely not a deck").run()
    next(b for b in at.button if b.label == "Generate presentation").click().run()
    assert not at.exception and at.error and "reference presentation" in at.error[0].value
    assert "Traceback" not in at.error[0].value


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_page_end_to_end_builtin_reference_shows_real_metrics():
    if not upload_ok(new_app(30).run()):
        raise SkipTest("installed Streamlit has no AppTest file_uploader support")
    at = new_app(300).run()
    at.file_uploader(key="paper").upload("My Paper.pdf", PAPER.read_bytes(), "application/pdf").run()
    next(b for b in at.button if b.label == "Generate presentation").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("Reference style: Academic Research (built-in)" in c.value for c in at.caption)
    _, res = pipeline_run("builtin")                                           # independent run of the same inputs
    ver = res.verification
    m = _metrics(at)
    for name, key in (("Dalloc", "dalloc"), ("Dorder", "dorder"), ("Dstyle", "dstyle"), ("Dpref", "dpref")):
        assert m[name] == f"{ver[key]:.4f}", (name, m[name], ver[key])
    assert m["Final Objective Score"] == f"{ver['objective']:.4f}" and m["Slides generated"] == str(res.n_slides)
    assert {"Source Support", "Numeric Fidelity", "Overflow Rate", "Preference Fidelity"} <= set(m)
    dl_labels = {getattr(b, "proto").label for b in at.get("download_button")}
    assert any("presentation" in (x or "").lower() and "pptx" in (x or "").lower() for x in dl_labels)
    assert any("verification" in (x or "").lower() for x in dl_labels)


def test_page_end_to_end_custom_reference():
    if not upload_ok(new_app(30).run()):
        raise SkipTest("installed Streamlit has no AppTest file_uploader support")
    at = new_app(300).run()
    at.checkbox(key="use_custom").check().run()
    at.file_uploader(key="paper").upload("My Paper.pdf", PAPER.read_bytes(), "application/pdf").run()
    at.file_uploader(key="custom_deck").upload("mine.pptx", DECK.read_bytes()).run()
    next(b for b in at.button if b.label == "Generate presentation").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("Reference style: Custom uploaded presentation" in c.value for c in at.caption)
    _, res = pipeline_run("custom", custom_bytes=DECK.read_bytes(), custom_name="mine.pptx")
    assert _metrics(at)["Dpref"] == f"{res.verification['dpref']:.4f}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import time
    skipped = []
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  ok  {t.__name__}  ({time.time() - t0:.1f}s)")
        except SkipTest as e:
            skipped.append(t.__name__)
            print(f"  SKIPPED  {t.__name__}: {e}")
    for staged, _ in _C.values():
        P.cleanup(staged.work_dir)
    n = len(tests) - len(skipped)
    print(f"ALL UI REFERENCE TESTS PASSED ({n} tests run, {len(skipped)} skipped)")

"""
Phase 4 tests (editable PPTX generation / template filling).   Run:   python tests/test_phase4.py
(also collectable by pytest).  Deterministic, no network.  Uses the Phase 2 plan and Phase 3 draft produced by the
real pipeline on the synthetic data (regenerated into a temp dir, so committed outputs are not touched).
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pptx import Presentation                                  # noqa: E402
from pptx.util import Pt                                       # noqa: E402

import pptx_generator as pg                                    # noqa: E402
from pptx_generator import expected_content_slides, generate_pptx, source_marker   # noqa: E402

PAPER = ROOT / "data" / "papers" / "synthetic_paper.pdf"
DECK = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
MAIN = ROOT / "src" / "main.py"

_CACHE = {}


def pipeline():
    """Run Phase 1 -> 2 -> 3 once through the CLI into a temp dir; return (plan, draft, tmpdir)."""
    if "p" not in _CACHE:
        tmp = Path(tempfile.mkdtemp(prefix="at_p4_"))
        cmd = [sys.executable, str(MAIN), "--paper", str(PAPER), "--deck", str(DECK), "--out", str(tmp / "p1"),
               "--plan", "--n-slides", "10", "--plan-out", str(tmp / "plan.json"),
               "--draft", "--draft-out", str(tmp / "draft.json")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        _CACHE["p"] = (json.loads((tmp / "plan.json").read_text(encoding="utf-8")),
                       json.loads((tmp / "draft.json").read_text(encoding="utf-8")), tmp)
    return _CACHE["p"]


def build(draft=None, plan=None, name="out.pptx", **kw):
    p, d, tmp = pipeline()
    out = tmp / name
    meta = generate_pptx(draft or d, plan or p, DECK, out, **kw)
    return out, meta


def slide_text(slide):
    return "\n".join(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)


def content_slides(prs):
    return [s for s in prs.slides if s.slide_layout.name == "Title and Content"
            and "Source Index" not in slide_text(s)]


# ------------------------------------------------------------------------------------------ tests #
def test_file_created_and_opens():
    out, meta = build()
    assert out.is_file() and out.stat().st_size > 10_000
    prs = Presentation(str(out))                                  # opens with python-pptx
    assert len(prs.slides) == meta["n_slides"]


def test_slide_count_matches_plan():
    plan, draft, _ = pipeline()
    out, meta = build()
    prs = Presentation(str(out))
    n_planned = sum(plan["slide_allocation"].values())
    assert n_planned == 10 == meta["n_content_slides"] == meta["n_content_slides_planned"]
    assert len(prs.slides) == 1 + n_planned + 1                   # title + content + Source Index
    out2, meta2 = build(name="nosrc.pptx", source_slide=False)
    assert len(Presentation(str(out2)).slides) == 1 + n_planned and meta2["n_slides"] == 11


def test_template_structure_preserved():
    tpl, out = Presentation(str(DECK)), Presentation(str(build()[0]))
    assert (out.slide_width, out.slide_height) == (tpl.slide_width, tpl.slide_height)
    assert [l.name for l in out.slide_layouts] == [l.name for l in tpl.slide_layouts]
    assert out.slides[0].slide_layout.name == "Title Slide"
    for s in list(out.slides)[1:]:
        assert s.slide_layout.name == "Title and Content"
        assert s.shapes.title.left == tpl.slide_layouts.get_by_name("Title and Content").placeholders[0].left
    # the reference paper's example slides (and their pictures/tables/charts) are not carried over
    assert not any(sh.shape_type in (13, 19, 3) for s in out.slides for sh in s.shapes)
    assert "A. Author" not in "".join(slide_text(s) for s in out.slides)


def test_section_titles_and_order():
    plan, draft, _ = pipeline()
    out, meta = build()
    prs = Presentation(str(out))
    exp = expected_content_slides(plan)
    body = list(prs.slides)[1:1 + len(exp)]
    for s, e in zip(body, exp):
        title = s.shapes.title.text_frame.text
        assert title.startswith(e["section"]), (title, e)
        if e["section_slide_count"] > 1:
            assert title.endswith(f"({e['section_slide_index']}/{e['section_slide_count']})")
    seen = []
    for e in exp:
        if e["section"] not in seen:
            seen.append(e["section"])
    assert seen == plan["section_order"] == meta["section_order"]
    assert prs.slides[0].shapes.title.text_frame.text == draft["title"]


def test_bullets_present_verbatim_and_budget_respected():
    plan, draft, _ = pipeline()
    out, meta = build()
    prs = Presentation(str(out))
    slides = content_slides(prs)
    assert len(slides) == len(draft["slides"])
    for s, d in zip(slides, draft["slides"]):
        body = [sh for sh in s.placeholders if sh.placeholder_format.idx == 1][0]
        paras = [p for p in body.text_frame.paragraphs if p.text.strip()]
        assert len(paras) == len(d["bullets"])                    # no padding, no extra bullets
        for p, b in zip(paras, d["bullets"]):
            assert p.text.startswith(b["text"])                   # verbatim Phase 3 text, not truncated
    assert meta["n_bullets"] == draft["n_bullets"] == meta["n_bullets_in_draft"]
    assert sum(meta["bullets_per_slide"].values()) == meta["n_bullets"]


def test_source_pointers_retained():
    plan, draft, _ = pipeline()
    out, meta = build()
    prs = Presentation(str(out))
    slides = content_slides(prs)
    n = 0
    for s, d in zip(slides, draft["slides"]):
        txt, notes = slide_text(s), s.notes_slide.notes_text_frame.text
        for b in d["bullets"]:
            m = source_marker(b["source"])
            assert m in txt and m in [r.text.strip() for sh in s.shapes if sh.has_text_frame
                                      for p in sh.text_frame.paragraphs for r in p.runs]
            assert b["source"]["sentence_id"] in notes and f"page {b['source']['page']}" in notes
            n += 1
    assert n == meta["n_source_pointers"] == draft["n_bullets"] and meta["all_bullets_have_source_pointer"]
    # marker sits in the SAME paragraph as its bullet, right after the bullet text
    first = slides[0].placeholders[1].text_frame.paragraphs[0]
    assert [r.text for r in first.runs][0] == draft["slides"][0]["bullets"][0]["text"]
    assert first.runs[1].text.strip() == source_marker(draft["slides"][0]["bullets"][0]["source"])
    # Source Index slide lists every pointer
    idx = slide_text(list(prs.slides)[-1])
    for d in draft["slides"]:
        for b in d["bullets"]:
            assert source_marker(b["source"]) in idx


def test_editable_text_no_images():
    out, _ = build()
    prs = Presentation(str(out))
    for s in prs.slides:
        assert not any(sh.shape_type == 13 for sh in s.shapes)    # no pictures
        text_shapes = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        assert len(text_shapes) >= 2
        assert all(sh.is_placeholder for sh in text_shapes)       # real template placeholders
    # round trip: text can be edited and saved again
    tmp = out.parent / "edited.pptx"
    prs.slides[1].shapes.title.text_frame.text = "Edited"
    prs.save(str(tmp))
    assert Presentation(str(tmp)).slides[1].shapes.title.text_frame.text == "Edited"


def test_no_empty_content_slides():
    out, _ = build()
    for i, s in enumerate(Presentation(str(out)).slides, 1):
        assert s.shapes.title.text_frame.text.strip(), i
        others = [sh for sh in s.shapes if sh.has_text_frame and sh is not s.shapes.title and sh.text_frame.text.strip()]
        assert others, f"slide {i} has no body text"
    # a planned slide with no bullets is skipped (documented), never emitted empty
    plan, draft, _ = pipeline()
    d2 = copy.deepcopy(draft)
    d2["slides"][1]["bullets"] = []
    out2, meta = build(draft=d2, name="skip.pptx")
    assert meta["n_content_slides"] == 9 and len(meta["skipped_empty_slides"]) == 1
    assert any("no bullets" in w for w in meta["warnings"])
    assert len(Presentation(str(out2)).slides) == 11


def test_overflow_adjustment_recorded_and_text_kept():
    plan, draft, _ = pipeline()
    d2 = copy.deepcopy(draft)
    long = " ".join(["Learned sorters exploit the structure of nearly sorted widget streams."] * 4)
    for b in d2["slides"][3]["bullets"]:
        b["text"] = long
    out, meta = build(draft=d2, name="overflow.pptx")
    adj = [o for o in meta["overflow_adjustments"] if o["slide"] == 5]
    assert len(adj) == 1 and adj[0]["font_reduced"] and adj[0]["fill_before"] > pg.MAX_FILL
    assert adj[0]["font_size_final_pt"] < adj[0]["font_size_start_pt"]
    assert adj[0]["font_size_final_pt"] >= pg.MIN_FONT_SIZE
    body = Presentation(str(out)).slides[4].placeholders[1]
    assert all(p.text.startswith(long) for p in body.text_frame.paragraphs if p.text.strip())   # nothing truncated
    for p in body.text_frame.paragraphs:
        assert p.runs[0].font.size is not None and p.runs[0].font.size >= Pt(pg.MIN_FONT_SIZE)


def test_min_font_configurable_and_unresolved_warned():
    plan, draft, _ = pipeline()
    d2 = copy.deepcopy(draft)
    huge = " ".join(["Learned sorters exploit the structure of nearly sorted widget streams."] * 30)
    d2["slides"][0]["bullets"][0]["text"] = huge
    out, meta = build(draft=d2, name="huge.pptx", min_font_size=24)
    rec = [o for o in meta["overflow_checks"] if o["slide"] == 2][0]
    assert rec["min_font_size_pt"] == 24 and rec["font_size_final_pt"] >= 24
    assert not rec["overflow_resolved"] and meta["n_unresolved_overflow"] >= 1
    assert any("overflow" in w for w in meta["warnings"])
    assert huge in Presentation(str(out)).slides[1].placeholders[1].text_frame.text      # still not truncated


def test_metadata_content_and_json_roundtrip():
    out, meta = build()
    meta2 = json.loads(json.dumps(meta))
    for k in ("output_pptx", "n_slides", "slide_to_section", "bullets_per_slide", "n_source_pointers",
              "overflow_adjustments", "overflow_checks", "warnings", "errors"):
        assert k in meta2
    assert meta2["output_pptx"] == str(out) and meta2["errors"] == []
    assert [m["section"] for m in meta2["slide_to_section"] if m["kind"] == "content"] == \
        [e["section"] for e in expected_content_slides(pipeline()[0])]
    assert meta2["n_overflow_adjustments"] == len(meta2["overflow_adjustments"])


def test_plan_draft_mismatch_is_an_error():
    plan, draft, _ = pipeline()
    p2 = copy.deepcopy(plan)
    p2["slide_allocation"]["Results"] += 1
    try:
        build(plan=p2, name="bad.pptx")
    except ValueError as e:
        assert "does not match the Phase 2 plan" in str(e)
    else:
        raise AssertionError("expected ValueError")
    try:
        generate_pptx(draft, plan, ROOT / "nope.pptx", pipeline()[2] / "x.pptx")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_deterministic_bytes():
    plan, draft, tmp = pipeline()
    import time
    a, _ = build(name="a.pptx")
    time.sleep(1.1)                       # zip entry timestamps have 2 s resolution: must not leak into the bytes
    b, _ = build(name="b.pptx")
    assert a.read_bytes() == b.read_bytes()


def test_cli_phase4_from_json_and_full_pipeline():
    plan, draft, tmp = pipeline()
    out = tmp / "cli" / "g.pptx"
    cmd = [sys.executable, str(MAIN), "--plan-json", str(tmp / "plan.json"), "--draft-json", str(tmp / "draft.json"),
           "--pptx", "--template", str(DECK), "--pptx-out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    meta = json.loads((out.parent / "pptx_metadata.json").read_text(encoding="utf-8"))
    assert meta["n_slides"] == 12 and out.is_file()
    # all four phases in one command
    out2 = tmp / "cli2" / "g.pptx"
    cmd = [sys.executable, str(MAIN), "--paper", str(PAPER), "--deck", str(DECK), "--out", str(tmp / "cli2" / "p1"),
           "--plan", "--n-slides", "10", "--plan-out", str(tmp / "cli2" / "plan.json"),
           "--draft", "--draft-out", str(tmp / "cli2" / "draft.json"), "--pptx", "--pptx-out", str(out2)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out2.read_bytes() == (build(name="ref.pptx")[0]).read_bytes()
    # clear error, non-zero exit
    bad = subprocess.run([sys.executable, str(MAIN), "--paper", str(PAPER), "--pptx"], capture_output=True, text=True)
    assert bad.returncode != 0 and "--pptx needs a Phase 3 draft" in bad.stderr


def test_earlier_phases_still_pass():
    from suite_runner import nested_run_by_parent, run_suites
    if nested_run_by_parent():        # test_phase5 is already running the Phase 1-3 suites itself, once each
        print("  (Phase 1-3 suites are being run by the parent suite; not repeated here)")
        return
    run_suites((1, 2, 3))             # standalone: run them concurrently, each with a timeout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import time
    for t in tests:
        t0 = time.time()
        t()
        print(f"  ok  {t.__name__}  ({time.time() - t0:.1f}s)")
    print(f"ALL PHASE 4 TESTS PASSED ({len(tests)} tests)")

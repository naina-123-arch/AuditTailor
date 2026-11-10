"""
Phase 5 tests (deterministic verification: Dalloc, Dorder, Dstyle, Dpref, PF, SR, NF, O, J).
Run:   python tests/test_phase5.py      (also collectable by pytest).   No network, no randomness.
Unit tests use tiny synthetic inputs; end-to-end tests run the real pipeline and tamper with the generated PPTX.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pptx import Presentation                       # noqa: E402
from pptx.util import Pt                            # noqa: E402

import verifier as V                                # noqa: E402
from suite_runner import single_thread_env          # noqa: E402

CLI_TIMEOUT = 240      # a hung pipeline subprocess is killed and reported instead of stalling the suite


def sh(cmd):
    """subprocess.run with captured text output, single-threaded numeric libs, and a hard timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, env=single_thread_env(), timeout=CLI_TIMEOUT)

PAPER = ROOT / "data" / "papers" / "synthetic_paper.pdf"
DECK = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
MAIN = ROOT / "src" / "main.py"
SECS = ["Motivation", "Background", "Method", "Setup", "Results", "Conclusion"]
close = lambda a, b, e=1e-9: abs(a - b) <= e


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# ----------------------------------------------------------------------------- shared pipeline run #
_C = {}


def pipe():
    if "x" not in _C:
        tmp = Path(tempfile.mkdtemp(prefix="at_p5_"))
        out = tmp / "deck.pptx"
        cmd = [sys.executable, str(MAIN), "--paper", str(PAPER), "--deck", str(DECK), "--out", str(tmp / "p1"),
               "--plan", "--n-slides", "10", "--plan-out", str(tmp / "plan.json"),
               "--draft", "--draft-out", str(tmp / "draft.json"), "--pptx", "--pptx-out", str(out),
               "--verify", "--verify-out", str(tmp / "ver.json")]
        proc = sh(cmd)
        assert proc.returncode == 0, proc.stderr
        L = lambda f: json.loads(Path(f).read_text(encoding="utf-8"))
        parsed = L(tmp / "p1" / "synthetic_paper_parsed.json")
        _C["x"] = dict(tmp=tmp, deck=out, parsed=parsed, qpp=L(tmp / "p1" / "synthetic_reference_qpp.json"),
                       plan=L(tmp / "plan.json"), draft=L(tmp / "draft.json"), ver=L(tmp / "ver.json"))
    return _C["x"]


def run_verify(deck, **kw):
    x = pipe()
    return V.verify(x["parsed"], x["qpp"], x["plan"], x["draft"], deck, **kw)


def tamper(fn, name):
    """Copy of the generated deck with ``fn(prs)`` applied."""
    x = pipe()
    prs = Presentation(str(x["deck"]))
    fn(prs)
    out = x["tmp"] / name
    prs.save(str(out))
    return out


# ------------------------------------------------------------------------------------ 1-2 Dalloc #
def test_dalloc_zero_when_identical():
    plan = {"Motivation": 2, "Method": 3, "Results": 1}
    v, d = V.dalloc(plan, dict(plan))
    assert v == 0.0 and d["K"] == 3 and d["absent_from_both"] == ["Background", "Setup", "Conclusion"]


def test_dalloc_increases_with_difference():
    plan = {"Motivation": 2, "Method": 3}
    v1, _ = V.dalloc(plan, {"Motivation": 2, "Method": 2})            # (0 + 1/3) / 2
    v2, _ = V.dalloc(plan, {"Motivation": 1, "Method": 1})            # (1/2 + 2/3) / 2
    assert close(v1, (1 / 3) / 2) and close(v2, (1 / 2 + 2 / 3) / 2) and 0 < v1 < v2 <= 1
    v3, d3 = V.dalloc({"Motivation": 2}, {"Motivation": 2, "Results": 5})    # planned 0, actual 5 -> term capped at 1
    assert close(v3, 0.5) and d3["terms"]["Results"]["term"] == 1.0


# ------------------------------------------------------------------------------------ 3-4 Dorder #
def test_dorder_zero_for_identical_order():
    v, d = V.dorder(SECS, list(SECS))
    assert v == 0.0 and d["discordant_pairs"] == []


def test_dorder_detects_reordering():
    one_swap, _ = V.dorder(SECS, ["Background", "Motivation"] + SECS[2:])
    assert close(one_swap, 1 / 15)
    rev, _ = V.dorder(SECS, SECS[::-1])
    assert close(rev, 1.0)
    # missing / extra sections: unrankable pairs count, nothing crashes
    miss, d = V.dorder(SECS, SECS[:-1])
    assert close(miss, 5 / 15) and d["missing_from_actual"] == ["Conclusion"]
    assert V.dorder([], [])[0] == 0.0 and V.dorder(["Method"], ["Method"])[0] == 0.0
    assert 0 < V.dorder(SECS, ["Method", "Results"])[0] <= 1.0


# ------------------------------------------------------------------------------------ 5-6 Dstyle #
PHI = {"eta": 0.5, "beta": 0.6, "rho": 0.3, "epsilon": 0.1}


def test_dstyle_zero_for_identical_qpps():
    v, d = V.dstyle(PHI, dict(PHI))
    assert v == 0.0 and d["features_included"] == ["eta", "beta", "rho", "epsilon"]
    x = pipe()
    assert V.dstyle(x["qpp"]["theta"]["phi"], x["qpp"]["theta"]["phi"])[0] == 0.0     # a real Phase 1 QPP


def test_dstyle_detects_changed_features():
    v, d = V.dstyle(PHI, {**PHI, "rho": 0.0})
    assert close(v, 0.3 / 4) and d["features"]["rho"]["abs_diff"] == 0.3
    v2, _ = V.dstyle(PHI, {"eta": 0.0, "beta": 0.0, "rho": 0.0, "epsilon": 0.0})
    assert v2 > v
    v3, _ = V.dstyle(PHI, {**PHI, "eta": 7.0})                          # eta can exceed 1: clamped
    assert close(v3, 1 / 4)
    v4, _ = V.dstyle(PHI, {**PHI, "rho": 0.0}, {"eta": 0, "beta": 0, "rho": 1, "epsilon": 0})   # configurable weights
    assert close(v4, 0.3)
    assert raises(ValueError, V.dstyle, PHI, PHI, {"eta": 0, "beta": 0, "rho": 0, "epsilon": 0})


# ------------------------------------------------------------------------------------ 7-8 Dpref/PF #
def test_dpref_from_components_and_pf():
    v, pf, d = V.dpref(0.5, 0.2, 0.1, {"alloc": 0.5, "order": 0.3, "style": 0.2})
    assert close(v, 0.25 + 0.06 + 0.02) and close(pf, 1 - v)
    v, pf, _ = V.dpref(0.5, 0.2, 0.1)                                   # defaults sum to 1
    assert close(v, 0.4 * 0.5 + 0.2 * 0.2 + 0.4 * 0.1) and close(pf, 1 - v)
    assert V.dpref(1, 1, 1)[:2] == (1.0, 0.0) and V.dpref(0, 0, 0)[:2] == (0.0, 1.0)
    assert raises(ValueError, V.dpref, 0.1, 0.1, 0.1, {"alloc": 0.5, "order": 0.5, "style": 0.5})
    assert raises(ValueError, V.dpref, 0.1, 0.1, 0.1, {"alloc": 1.2, "order": -0.2, "style": 0.0})


# ------------------------------------------------------------------------------------ 9-10 SR #
PARSED = {"sentences": [
    {"id": 0, "page": 1, "text": "Runtime drops by 31 percent relative to quicksort on nearly sorted inputs."},
    {"id": 1, "page": 1, "text": "The predictor has two hidden layers and a softmax output over buckets."},
    {"id": 2, "page": 2, "text": "Accuracy rises from 0.76 to 0.87 when the predictor is enabled."}]}


def B(i, text, sid, page=1, slide=2):
    return {"id": f"s{slide}.b{i}", "slide": slide, "text": text, "sentence_id": sid, "page": page}


def test_sr_identical_extractive_bullets_supported():
    bl = [B(1, PARSED["sentences"][0]["text"], 0), B(2, PARSED["sentences"][1]["text"], 1)]
    r = V.source_support(bl, PARSED)
    assert (r["supported_bullets"], r["total_bullets"], r["sr"]) == (2, 2, 1.0) and r["unsupported_ids"] == []
    assert all(p["cosine"] > 0.999 for p in r["per_bullet"])
    assert "NOT semantic entailment" in r["method"]
    # clause-trimmed (subset) bullet still supported at the default threshold
    sub = V.source_support([B(1, "Runtime drops by 31 percent relative to quicksort.", 0)], PARSED)
    assert sub["supported_bullets"] == 1


def test_sr_unsupported_text_detected():
    bl = [B(1, PARSED["sentences"][0]["text"], 0),
          B(2, "Quantum entanglement guarantees perfect bucket prediction across galaxies.", 1),       # unrelated
          B(3, "Runtime drops by 31 percent.", None),                                                  # no pointer
          B(4, "Runtime drops by 31 percent relative to quicksort.", 99),                              # dangling pointer
          B(5, PARSED["sentences"][2]["text"], 2, page=1)]                                             # wrong page
    r = V.source_support(bl, PARSED)
    assert r["supported_bullets"] == 1 and r["total_bullets"] == 5 and close(r["sr"], 0.2)
    assert r["unsupported_ids"] == ["s2.b2", "s2.b3", "s2.b4", "s2.b5"]
    reasons = {u["id"]: u["reason"] for u in r["unsupported"]}
    assert "cosine" in reasons["s2.b2"] and "no source pointer" in reasons["s2.b3"]
    assert "does not exist" in reasons["s2.b4"] and "page" in reasons["s2.b5"]
    # threshold is configurable
    assert V.source_support(bl[:2], PARSED, threshold=0.0)["supported_bullets"] == 2
    assert raises(ValueError, V.source_support, bl, PARSED, 1.5)
    # appended invented words lower the score of an otherwise identical sentence
    padded = V.source_support([B(1, PARSED["sentences"][0]["text"] + " Furthermore zebras orchestrate quantum "
                                    "blockchain synergies across distributed marketplaces entirely.", 0)], PARSED)
    assert padded["per_bullet"][0]["cosine"] < 0.7 and padded["supported_bullets"] == 0


# ------------------------------------------------------------------------------------ 11-12 NF #
def test_nf_matching_numbers():
    bl = [B(1, "Accuracy rises from 0.76 to 0.87.", 2), B(2, "Runtime drops by 31 percent.", 0),
          B(3, "The predictor has two hidden layers.", 1)]                                  # no numeric content
    r = V.numeric_fidelity(bl, PARSED)
    assert (r["numeric_claims_checked"], r["numeric_claims_correct"], r["nf"]) == (3, 3, 1.0)
    assert r["bullets_with_numbers"] == 2 and r["incorrect"] == []
    assert V.extract_numbers("1,000 and 0.80 and L2 and S004 and 31%") == ["1000", "0.8", "31"]
    empty = V.numeric_fidelity([B(1, "No numbers here.", 1)], PARSED)
    assert empty["nf"] == 1.0 and empty["vacuous"] and empty["numeric_claims_checked"] == 0


def test_nf_detects_changed_numbers():
    bl = [B(1, "Accuracy rises from 0.76 to 0.97.", 2),             # 0.87 -> 0.97
          B(2, "Runtime drops by 31 percent.", 0),
          B(3, "Runtime drops by 13 percent and 5 times.", 0)]     # 13 and 5 are not in the source
    r = V.numeric_fidelity(bl, PARSED)
    assert (r["numeric_claims_checked"], r["numeric_claims_correct"]) == (5, 2) and close(r["nf"], 0.4)
    assert [i["id"] for i in r["incorrect"]] == ["s2.b1", "s2.b3"] and r["incorrect"][0]["incorrect"] == ["0.97"]
    # a number cannot be claimed twice when the source states it once
    assert V.numeric_fidelity([B(1, "It dropped 31 and 31 percent.", 0)], PARSED)["numeric_claims_correct"] == 1
    # unresolvable pointer: numbers cannot be verified -> incorrect, not a crash
    assert V.numeric_fidelity([B(1, "It is 31.", 99)], PARSED)["nf"] == 0.0


# ------------------------------------------------------------------------------------ 13 Overflow #
def test_overflow_rate_calculation():
    assert V.overflow_from_counts(0, 10) == 0.0 and close(V.overflow_from_counts(3, 12), 0.25)
    assert V.overflow_from_counts(0, 0) == 0.0 and V.overflow_from_counts(5, 5) == 1.0
    x = pipe()
    clean = V.overflow_rate(x["deck"])
    assert clean["overflow_count"] == 0 and clean["overflow_rate"] == 0.0 and clean["total_checked"] == 24

    def blow(prs):                                       # slide 5: huge text, and slide 3: tiny font
        body = prs.slides[4].placeholders[1].text_frame
        body.paragraphs[0].runs[0].text = "Learned sorters exploit structure. " * 40
        for r in prs.slides[2].placeholders[1].text_frame.paragraphs[0].runs[:1]:
            r.font.size = Pt(10)
    bad = V.overflow_rate(tamper(blow, "overflow.pptx"))
    assert bad["overflow_count"] == 2 and bad["total_checked"] == 24 and close(bad["overflow_rate"], 2 / 24)
    assert bad["affected_slides"] == [3, 5] and len(bad["affected_boxes"]) == 2
    reasons = {b["slide"]: b["reasons"][0] for b in bad["boxes"] if b["overflow"]}
    # Primary detector reports actual PPTX height overflow (not just estimate_fill)
    assert ("pptx text height" in reasons[5] or "estimate_fill" in reasons[5]) and "font 10 pt" in reasons[3]
    blown_box = next(b for b in bad["boxes"] if b["slide"] == 5 and b["overflow"])
    assert blown_box.get("actual_overflow") is True
    assert blown_box.get("needed_pt") is not None and blown_box["needed_pt"] > blown_box["available_pt"]
    # source markers are small by design and must not count as min-font violations
    assert not any(b["overflow"] for b in clean["boxes"] if b["slide"] == 2)
    # configurable minimum
    assert V.overflow_rate(x["deck"], min_font_size=40)["overflow_count"] > 0


# ------------------------------------------------------------------------------------ 14 Objective #
def test_objective_formula():
    j, d = V.objective(0.1, 0.25, 0.8, 0.5, {"mu": 2.0, "nu": 3.0, "xi": 4.0})
    assert close(j, 0.1 + 2 * 0.25 + 3 * 0.2 + 4 * 0.5)
    assert close(d["components"]["xi_1_minus_SR"], 2.0) and "lower is better" in d["formula"]
    j, _ = V.objective(0.2, 0.0, 1.0, 1.0)
    assert close(j, 0.2)                                                  # perfect faithfulness, no overflow
    j, _ = V.objective(0.1, 0.5, 0.5, 0.5)                                # defaults mu=.5 nu=1 xi=1
    assert close(j, 0.1 + 0.25 + 0.5 + 0.5)
    assert raises(ValueError, V.objective, 0, 0, 1, 1, {"mu": 1, "nu": 1})


# ------------------------------------------------------------------------------------ 15 Edge cases #
def test_missing_and_empty_sections_do_not_crash():
    assert V.dalloc({}, {})[0] == 0.0
    assert V.dalloc({"Motivation": 0, "Results": 0}, {})[0] == 0.0
    v, _ = V.dalloc({"Motivation": 2, "Setup": 0}, {})                    # everything missing
    assert v == 1.0
    assert V.dorder([], SECS)[0] == 1.0 and V.dorder(SECS, [])[0] == 1.0
    assert V.source_support([], PARSED)["sr"] == 1.0 and V.source_support([], PARSED)["vacuous"]
    assert V.numeric_fidelity([], {"sentences": []})["nf"] == 1.0
    assert V.source_support([B(1, "text", 0)], {"sentences": []})["supported_bullets"] == 0
    # end-to-end: a deck whose content slides are all removed
    def strip(prs):
        lst = prs.slides._sldIdLst
        for sld in list(lst)[1:]:
            prs.part.drop_rel(sld.rId)
            lst.remove(sld)
    r = run_verify(tamper(strip, "empty.pptx"))
    assert r["dalloc"] == 1.0 and r["dorder"] == 1.0 and r["sr"] == 1.0 and r["details"]["sr"]["vacuous"]
    assert 0 <= r["pf"] <= 1 and any("no content slides" in w for w in r["warnings"])
    # a plan with an absent section and a one-section deck
    x = pipe()
    plan = {"slide_allocation": {"Motivation": 10}, "section_order": ["Motivation"]}
    r2 = V.verify(x["parsed"], x["qpp"], plan, x["draft"], x["deck"])
    assert 0 <= r2["dalloc"] <= 1 and 0 <= r2["objective"]


# ------------------------------------------------------------------------------- end-to-end checks #
def test_verification_json_from_real_pipeline():
    x = pipe()
    v = x["ver"]
    assert (x["tmp"] / "ver.json").is_file() and v["schema"] == V.SCHEMA
    for k in ("dalloc", "dorder", "dstyle", "dpref", "pf", "sr", "nf", "overflow_rate", "objective", "details", "config"):
        assert k in v
    assert v["dalloc"] == 0.0 and v["dorder"] == 0.0                      # Phase 4 follows the plan exactly
    assert v["sr"] == 1.0 and v["nf"] == 1.0 and v["overflow_rate"] == 0.0
    assert 0 < v["dstyle"] < 1 and close(v["pf"], 1 - v["dpref"], 1e-5)
    assert close(v["dpref"], 0.4 * v["dalloc"] + 0.2 * v["dorder"] + 0.4 * v["dstyle"], 1e-5)
    assert close(v["objective"], v["dpref"] + 0.5 * v["overflow_rate"] + (1 - v["nf"]) + (1 - v["sr"]), 1e-5)
    d = v["details"]
    assert d["sr"]["total_bullets"] == x["draft"]["n_bullets"] == 20 and d["nf"]["numeric_claims_checked"] == 4
    assert d["generated_qpp"]["excluded_slides"] == [12]                  # Source Index not counted as content
    assert d["generated_qpp"]["section_slide_counts"] == x["plan"]["slide_allocation"]


def test_tampered_deck_is_penalised():
    x = pipe()
    base = run_verify(x["deck"])

    def edit(prs):
        for r in prs.slides[6].placeholders[1].text_frame.paragraphs[0].runs[:1]:   # slide 7 (Setup)
            r.text = "We evaluate on seven synthetic datasets of two million widgets each, using zebra clusters."
        # slide 8 bullet (Results) -> wrong number
        r = prs.slides[7].placeholders[1].text_frame.paragraphs[0].runs[0]
        r.text = r.text.replace("0.76", "0.67")
    r = run_verify(tamper(edit, "tampered.pptx"))
    assert r["sr"] < base["sr"] and r["nf"] < 1.0 and r["objective"] > base["objective"]
    assert r["details"]["sr"]["unsupported_ids"] and r["details"]["nf"]["incorrect"]


def test_reordered_deck_changes_order_and_allocation():
    def swap(prs):                                       # rename slide 2 (Motivation) as Results, slide 11 as Motivation
        prs.slides[1].shapes.title.text_frame.text = "Experimental Results"
        prs.slides[10].shapes.title.text_frame.text = "Introduction"
    r = run_verify(tamper(swap, "reordered.pptx"))
    assert r["dalloc"] > 0 and r["dorder"] > 0 and r["pf"] < pipe()["ver"]["pf"]


def test_deterministic_and_cli_errors():
    x = pipe()
    a, b = run_verify(x["deck"]), run_verify(x["deck"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["dpref"] == x["ver"]["dpref"] and a["objective"] == x["ver"]["objective"]
    # Phase 5 only, from JSON files + a PPTX
    out = x["tmp"] / "only5.json"
    cmd = [sys.executable, str(MAIN), "--parsed", str(x["tmp"] / "p1" / "synthetic_paper_parsed.json"),
           "--qpp", str(x["tmp"] / "p1" / "synthetic_reference_qpp.json"), "--plan-json", str(x["tmp"] / "plan.json"),
           "--draft-json", str(x["tmp"] / "draft.json"), "--generated-pptx", str(x["deck"]), "--verify",
           "--verify-out", str(out), "--dpref-weights", "0.5", "0.25", "0.25", "--objective-weights", "1", "1", "1"]
    proc = sh(cmd)
    assert proc.returncode == 0, proc.stderr
    w = json.loads(out.read_text(encoding="utf-8"))
    assert w["config"]["pref_weights"] == {"alloc": 0.5, "order": 0.25, "style": 0.25}
    assert close(w["dpref"], 0.25 * w["dstyle"], 1e-5)
    bad = sh(cmd[:-8] + ["--dpref-weights", "0.5", "0.5", "0.5"])
    assert bad.returncode == 1 and "sum to 1" in bad.stderr
    noteck = sh([sys.executable, str(MAIN), "--paper", str(PAPER), "--verify"])
    assert noteck.returncode != 0 and "--verify needs" in noteck.stderr


# ------------------------------------------------------------------------------------ 16-19 earlier #
def test_earlier_phases_still_pass():
    from suite_runner import nested_run_by_parent, run_suites
    if nested_run_by_parent():        # test_phase6 is already running the Phase 1-5 suites itself, once each
        print("  (Phase 1-4 suites are being run by the parent suite; not repeated here)")
        return
    run_suites((1, 2, 3, 4))          # standalone: Phase 1-4 suites run concurrently, each with a timeout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import time
    for t in tests:
        t0 = time.time()
        t()
        print(f"  ok  {t.__name__}  ({time.time() - t0:.1f}s)")
    print(f"ALL PHASE 5 TESTS PASSED ({len(tests)} tests)")

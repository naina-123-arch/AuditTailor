"""
Phase 6 tests (bounded, deterministic, extractive repair loop).
Run:   python tests/test_phase6.py      (also collectable by pytest).   No network, no randomness, no LLM.

The repair loop is tested on REAL failures: the clean pipeline deck is tampered with (overflow, unsupported text,
altered numbers, wrong section labels / order / allocation) and Phase 6 must detect and repair the right thing.
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

from pptx import Presentation                       # noqa: E402
from pptx.util import Pt                            # noqa: E402

import repair as R                                  # noqa: E402
import verifier as V                                # noqa: E402
from suite_runner import run_suites, single_thread_env   # noqa: E402

PAPER = ROOT / "data" / "papers" / "synthetic_paper.pdf"
DECK = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
MAIN = ROOT / "src" / "main.py"
CLI_TIMEOUT = 240
close = lambda a, b, e=1e-9: abs(a - b) <= e

# ----------------------------------------------------------------------------- shared pipeline run #
_C = {}


def pipe():
    """ONE real end-to-end run (Phases 1-6 through the CLI); everything else reuses its artefacts."""
    if "x" not in _C:
        tmp = Path(tempfile.mkdtemp(prefix="at_p6_"))
        cmd = [sys.executable, str(MAIN), "--paper", str(PAPER), "--deck", str(DECK), "--out", str(tmp / "p1"),
               "--plan", "--n-slides", "10", "--plan-out", str(tmp / "plan.json"),
               "--draft", "--draft-out", str(tmp / "draft.json"), "--pptx", "--pptx-out", str(tmp / "deck.pptx"),
               "--verify-out", str(tmp / "ver.json"), "--repair", "--repair-out", str(tmp / "phase6")]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=single_thread_env(), timeout=CLI_TIMEOUT)
        assert proc.returncode == 0, proc.stderr
        L = lambda f: json.loads(Path(f).read_text(encoding="utf-8"))
        _C["x"] = dict(tmp=tmp, deck=tmp / "deck.pptx", parsed=L(tmp / "p1" / "synthetic_paper_parsed.json"),
                       qpp=L(tmp / "p1" / "synthetic_reference_qpp.json"), plan=L(tmp / "plan.json"),
                       draft=L(tmp / "draft.json"), ver=L(tmp / "ver.json"), stdout=proc.stdout)
    return _C["x"]


def tamper(fn, name):
    x = pipe()
    prs = Presentation(str(x["deck"]))
    fn(prs)
    out = x["tmp"] / name
    prs.save(str(out))
    return out


def repair(deck, name, draft=None, cfg=None, repairers=None):
    x = pipe()
    return R.repair_loop(x["parsed"], x["qpp"], x["plan"], draft or x["draft"], deck, x["tmp"] / name, template=DECK,
                         config=cfg, repairers=repairers)


def verify(deck, draft=None):
    x = pipe()
    return V.verify(x["parsed"], x["qpp"], x["plan"], draft or x["draft"], deck)


def deck_texts(path):
    prs = Presentation(str(path))
    return [[p.text for sh in s.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs] for s in prs.slides]


# ----------------------------------------------------------------------------------- tamper recipes #
def t_small_font(slides):
    def f(prs):
        for i in slides:
            for r in prs.slides[i].placeholders[1].text_frame.paragraphs[0].runs[:1]:
                r.font.size = Pt(10)
    return f


def t_blow_up(prs):                                       # slide 5: huge text
    prs.slides[4].placeholders[1].text_frame.paragraphs[0].runs[0].text = "Learned sorters exploit structure. " * 40


def t_unsupported(prs):                                   # slide 7 (Setup) bullet 1 -> unrelated invented text
    prs.slides[6].placeholders[1].text_frame.paragraphs[0].runs[0].text = \
        "We evaluate on seven synthetic datasets of two million widgets each, using zebra clusters."


def t_number(prs):                                        # slide 8 (Results): 0.76 -> 0.67
    r = prs.slides[7].placeholders[1].text_frame.paragraphs[0].runs[0]
    r.text = r.text.replace("0.76", "0.67")


def t_titles(prs):                                        # wrong section labels on two slides
    prs.slides[1].shapes.title.text_frame.text = "Experimental Results"
    prs.slides[10].shapes.title.text_frame.text = "Introduction"


def t_reorder(prs):                                       # content slide 5 moved to position 3
    lst = prs.slides._sldIdLst
    el = list(lst)[4]
    lst.remove(el)
    lst.insert(2, el)


def t_drop(prs):                                          # one planned content slide missing (allocation drift)
    lst = prs.slides._sldIdLst
    sld = list(lst)[5]
    prs.part.drop_rel(sld.rId)
    lst.remove(sld)


def bullet_text(path, sid_slide, k=1):
    return next(b["text"] for b in V.read_deck_bullets(path) if b["slide"] == sid_slide and b["id"].endswith(f".b{k}"))


# ------------------------------------------------------------------------------------ 1 config #
def test_thresholds_config_and_priority_detection():
    c = R.RepairConfig()
    assert (c.max_rounds, c.max_overflow_rate, c.min_support_ratio, c.min_numeric_fidelity) == \
           (R.MAX_REPAIR_ROUNDS, R.MAX_OVERFLOW_RATE, R.MIN_SUPPORT_RATIO, R.MIN_NUMERIC_FIDELITY) == (3, 0.0, 1.0, 1.0)
    assert R.FAMILIES == ("overflow", "support", "preference")                     # explicit repair priority
    ver = {"overflow_rate": 0.1, "sr": 0.5, "nf": 0.5, "dalloc": 0.2, "dorder": 0.1, "dpref": 0.9}
    f = R.detect_failures(ver, c)
    assert [x["family"] for x in f] == ["overflow", "support", "support", "preference", "preference", "preference"]
    assert R.detect_failures({"overflow_rate": 0, "sr": 1, "nf": 1, "dalloc": 0, "dorder": 0, "dpref": 0.1}, c) == []
    assert R.detect_failures({**ver, "dpref": 0.0, "dalloc": 0, "dorder": 0, "sr": 1, "nf": 1, "overflow_rate": 0.0}, c) == []
    relaxed = R.RepairConfig(max_overflow_rate=0.5, min_support_ratio=0.4, min_numeric_fidelity=0.4,
                             max_preference_distance=1.0, max_dalloc=1.0, max_dorder=1.0)
    assert R.detect_failures(ver, relaxed) == []                                   # thresholds are configurable
    assert raises(ValueError, R.RepairConfig, max_rounds=-1) and raises(ValueError, R.RepairConfig, min_support_ratio=2)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# ------------------------------------------------------------------------------ 2 clean deck #
def test_clean_deck_is_not_modified():
    x = pipe()
    res = repair(x["deck"], "clean")
    h = res["history"]
    assert h["rounds_used"] == 0 and h["stop_reason"] == "all_criteria_satisfied" and h["rounds"] == []
    assert h["final"]["improved_over_initial"] is False and h["final"]["selected_round"] == 0
    assert (x["tmp"] / "clean" / "final.pptx").read_bytes() == x["deck"].read_bytes()      # byte-identical
    assert res["verification"]["objective"] == x["ver"]["objective"] == h["initial"]["J"]


# -------------------------------------------------------------------------- 3 termination / bound #
def test_loop_terminates_and_respects_max_rounds():
    deck = tamper(t_small_font([2, 3, 4]), "three_small.pptx")
    assert verify(deck)["details"]["overflow"]["overflow_count"] == 3

    def one_box_only(src, dst, boxes, mf):                      # a deliberately slow overflow repairer: 1 box per call
        return R.repair_overflow(src, dst, boxes[:1], mf)
    for limit in (0, 1, 2):
        res = repair(deck, f"bound{limit}", cfg=R.RepairConfig(max_rounds=limit), repairers={"overflow": one_box_only})
        h = res["history"]
        assert h["rounds_used"] == limit == len(h["rounds"]) <= h["max_repair_rounds"]
        assert h["stop_reason"] == "max_repair_rounds_reached"
    assert res["verification"]["details"]["overflow"]["overflow_count"] == 1                 # 3 -> 1 after 2 rounds
    res3 = repair(deck, "bound3", cfg=R.RepairConfig(max_rounds=3), repairers={"overflow": one_box_only})
    assert res3["history"]["rounds_used"] == 3 and res3["verification"]["overflow_rate"] == 0.0
    assert res3["history"]["stop_reason"] == "all_criteria_satisfied"
    assert R.MAX_REPAIR_ROUNDS == 3
    assert repair(deck, "default_bound")["history"]["rounds_used"] <= 3


# -------------------------------------------------------------------------------- 4 overflow #
def test_overflow_repair_reduces_overflow_without_deleting_text():
    x = pipe()
    deck = tamper(t_small_font([2]), "small.pptx")
    before = verify(deck)
    assert before["overflow_rate"] > 0 and before["details"]["overflow"]["affected_slides"] == [3]
    res = repair(deck, "ovf")
    after, h = res["verification"], res["history"]
    assert after["overflow_rate"] == 0.0 and h["final"]["improved_over_initial"] and after["objective"] < before["objective"]
    assert deck_texts(res["pptx"]) == deck_texts(deck)                         # not a single character deleted or changed
    assert [a["repair_type"] for a in h["rounds"][0]["actions"]] == ["overflow"]
    fix = h["rounds"][0]["actions"][0]["what"]["boxes"][0]
    assert fix["slide"] == 3 and fix["runs_raised_to_minimum"] >= 1 and fix["font_after_pt"] >= 16
    # genuinely huge text cannot be shrunk away; overflow repair alone must NOT delete it (support repair handles it)
    big = tamper(t_blow_up, "big.pptx")
    only = R.repair_overflow(big, x["tmp"] / "big_fit.pptx", [b for b in verify(big)["details"]["overflow"]["boxes"] if b["overflow"]], 16)
    assert only["changed"] and deck_texts(x["tmp"] / "big_fit.pptx") == deck_texts(big)


def test_dense_results_gets_continuation_slide_not_shortening():
    """
    Regression: a deliberately dense Results slide must be repaired by
    (1) reasonable font/spacing fit, then (2) continuation slide(s).
    No bullets deleted, no numbers changed, no source pointers dropped, O=0 after.
    Normal slides that already fit must not receive unnecessary continuations.
    """
    from pptx.util import Pt
    import pptx_generator as G

    x = pipe()
    prs = Presentation(str(x["deck"]))
    # Results (1/3) is slide index 7 in the synthetic pipeline deck
    body = prs.slides[7].placeholders[1]
    dense = [
        ("Accuracy rises from 0.76 to 0.87 when the predictor is enabled on the primary set.", "[S015, p.1]"),
        ("Throughput improves by 2.3× under the learned sorting policy versus the static baseline.", "[S016, p.2]"),
        ("Latency at the 99th percentile drops from 48 ms to 19 ms once adaptive batching is on.", "[S017, p.2]"),
        ("Memory footprint stays within 4% of baseline while accuracy gains are retained.", "[S018, p.3]"),
        ("Ablation without the ranking head reduces the accuracy lift to only 0.03 points.", "[S019, p.3]"),
        ("Cross-domain transfer preserves 91% of the original gain without fine-tuning.", "[S020, p.4]"),
        ("Human preference for the new ranking is 68% versus 32% for the previous model.", "[S021, p.4]"),
        ("Training wall-clock increases by less than 12% when the ranking loss is added.", "[S022, p.5]"),
        ("Calibration error (ECE) falls from 0.09 to 0.04 after temperature scaling.", "[S023, p.5]"),
        ("End-to-end A/B test on 3.2M sessions shows a +1.8% relative conversion lift.", "[S024, p.6]"),
        ("Feature importance ranks the temporal context embedding first among inputs.", "[S025, p.6]"),
        ("Sensitivity analysis: 10% less training data still yields accuracy above 0.84.", "[S026, p.7]"),
    ]
    G._fill_body(body, [{"text": t, "marker": m, "level": 0} for t, m in dense])
    for p in body.text_frame.paragraphs:
        for r in p.runs:
            if r.text.strip() and not r.text.strip().startswith("["):
                r.font.size = Pt(18)
    dense_path = x["tmp"] / "dense_results.pptx"
    prs.save(str(dense_path))

    before = V.overflow_rate(dense_path, min_font_size=14.0)
    assert before["overflow_count"] >= 1, "dense slide must overflow before repair"
    boxes = [b for b in before["boxes"] if b.get("overflow")]
    out = x["tmp"] / "dense_results_repaired.pptx"
    rec = R.repair_overflow(dense_path, out, boxes, min_font=14.0)
    assert rec["changed"]
    assert any(bx.get("split") for bx in rec["boxes"]), "continuation split must be used"
    after = V.overflow_rate(out, min_font_size=14.0)
    assert after["overflow_count"] == 0 and after["overflow_rate"] == 0.0

    # All original main texts and markers preserved; no numerical change
    import re
    def extract(path):
        prs2 = Presentation(str(path))
        texts = []
        for s in prs2.slides:
            for sh in s.shapes:
                if not sh.has_text_frame:
                    continue
                try:
                    if sh.is_placeholder and sh.placeholder_format.type in (1, 3):
                        continue
                except Exception:
                    pass
                for p in sh.text_frame.paragraphs:
                    if p.text.strip():
                        texts.append(p.text)
        return texts
    tb, ta = extract(dense_path), extract(out)
    for main, _mk in dense:
        assert any(main in t for t in ta), f"bullet lost: {main[:50]}"
    marks_b = set(re.findall(r"\[S\d+,\s*p\.\d+\]", " ".join(tb)))
    marks_a = set(re.findall(r"\[S\d+,\s*p\.\d+\]", " ".join(ta)))
    assert marks_b <= marks_a
    for num in ("0.76", "0.87", "2.3×", "0.03", "91%", "0.09", "0.04", "1.8%", "0.84"):
        assert num in " ".join(ta)

    # Continuation title format
    prs_out = Presentation(str(out))
    titles = [s.shapes.title.text_frame.text for s in prs_out.slides if s.shapes.title]
    assert any("— continued" in t for t in titles)
    assert any(t.startswith("Results (1/3)") for t in titles)

    # Normal deck: no unnecessary continuation when nothing overflows
    clean = x["deck"]
    ov_clean = V.overflow_rate(clean, min_font_size=14.0)
    if ov_clean["overflow_count"] == 0:
        rec_clean = R.repair_overflow(clean, x["tmp"] / "clean_noop.pptx", ov_clean["boxes"][:1], 14.0)
        assert not any(bx.get("split") for bx in rec_clean["boxes"])
        assert len(Presentation(str(x["tmp"] / "clean_noop.pptx")).slides) == len(Presentation(str(clean)).slides)


# ---------------------------------------------------------------------------- 5-6 support / NF #
def test_support_repair_restores_source_support():
    x = pipe()
    deck = tamper(t_unsupported, "unsup.pptx")
    before = verify(deck)
    assert before["sr"] < 1.0 and "s7.b1" in before["details"]["sr"]["unsupported_ids"]
    res = repair(deck, "sup")
    after = res["verification"]
    assert after["sr"] == 1.0 and after["nf"] == 1.0 and after["objective"] < before["objective"]
    assert bullet_text(res["pptx"], 7) == bullet_text(x["deck"], 7)            # exactly the Phase 3 bullet again
    b7 = next(b for b in V.read_deck_bullets(res["pptx"]) if b["id"] == "s7.b1")
    assert b7["marker"] == next(b for b in V.read_deck_bullets(x["deck"]) if b["id"] == "s7.b1")["marker"]   # pointer kept
    act = res["history"]["rounds"][0]["actions"][0]
    assert act["repair_type"] == "support_fidelity" and act["what"]["bullets"][0]["restored_from"] == "phase3_draft_text"


def test_numeric_repair_restores_numeric_fidelity():
    x = pipe()
    deck = tamper(t_number, "num.pptx")
    before = verify(deck)
    assert before["nf"] < 1.0 and before["details"]["nf"]["incorrect"][0]["incorrect"] == ["0.67"]
    res = repair(deck, "num")
    after = res["verification"]
    assert after["nf"] == 1.0 and after["sr"] == 1.0 and "0.67" not in bullet_text(res["pptx"], 8)
    assert bullet_text(res["pptx"], 8) == bullet_text(x["deck"], 8)
    assert res["history"]["rounds"][0]["actions"][0]["what"]["bullets"][0]["why"] == ["nf"] or \
        "nf" in res["history"]["rounds"][0]["actions"][0]["what"]["bullets"][0]["why"]


def test_draft_fallback_uses_original_source_sentence_not_invention():
    x = pipe()
    draft = copy.deepcopy(x["draft"])
    setup = next(s for s in draft["slides"] if s["section"] == "Setup")
    orig = setup["bullets"][0]["original_text"]
    setup["bullets"][0]["text"] = "Zebras orchestrate quantum blockchain synergies across marketplaces."   # corrupt the DRAFT too
    deck = tamper(t_unsupported, "unsup2.pptx")
    assert verify(deck, draft)["sr"] < 1.0
    res = repair(deck, "fallback", draft=draft)
    assert res["verification"]["sr"] == 1.0 and bullet_text(res["pptx"], 7) == orig
    new_b = next(s for s in res["draft"]["slides"] if s["section"] == "Setup")["bullets"][0]
    assert new_b["text"] == orig and any(t.startswith("repair:") for t in new_b["trim_ops"])
    assert draft["slides"] != res["draft"]["slides"] and setup["bullets"][0]["text"].startswith("Zebras")   # input not mutated


# ------------------------------------------------------------------------------- 7 preference #
def test_preference_repair_fixes_labels_order_and_allocation():
    x = pipe()
    # (a) wrong section labels
    deck = tamper(t_titles, "titles.pptx")
    b = verify(deck)
    assert b["dalloc"] > 0 and b["dorder"] > 0
    res = repair(deck, "pref_titles")
    a = res["verification"]
    assert a["dalloc"] == 0.0 and a["dorder"] == 0.0 and a["objective"] < b["objective"]
    assert res["history"]["rounds"][0]["actions"][0]["what"]["mode"] == "in_place"
    assert deck_texts(res["pptx"])[1] == deck_texts(x["deck"])[1]
    # (b) reordered slides (content follows its own pointers; title + index follow the plan)
    deck = tamper(t_reorder, "reorder.pptx")
    b = verify(deck)
    assert b["dorder"] > 0
    res = repair(deck, "pref_order")
    assert res["verification"]["dorder"] == 0.0 and res["verification"]["dalloc"] == 0.0
    assert [r["text"] for r in V.read_deck_bullets(res["pptx"])] == [r["text"] for r in V.read_deck_bullets(x["deck"])]
    assert res["history"]["rounds"][0]["actions"][0]["what"]["moved"]
    # (c) a planned slide is missing -> allocation drift -> rebuilt from plan + draft (no new content)
    deck = tamper(t_drop, "dropped.pptx")
    b = verify(deck)
    assert b["dalloc"] > 0
    res = repair(deck, "pref_rebuild")
    a = res["verification"]
    assert a["dalloc"] == 0.0 and a["dorder"] == 0.0 and a["sr"] == 1.0 and a["objective"] < b["objective"]
    assert res["history"]["rounds"][0]["actions"][0]["what"]["mode"] == "rebuild_from_plan_and_draft"
    assert a["details"]["generated_qpp"]["section_slide_counts"] == x["plan"]["slide_allocation"]
    # reference QPP and plan are never touched
    assert x["plan"]["slide_allocation"] == pipe()["plan"]["slide_allocation"]


# ----------------------------------------------------------------------------- 8 non-improving #
def test_non_improving_candidate_is_not_selected():
    x = pipe()
    deck = tamper(t_number, "num_bad.pptx")

    def harmful(src, dst, ver, parsed, draft, plan, thr):        # a "repair" that makes the deck worse
        prs = Presentation(str(src))
        p = prs.slides[7].placeholders[1].text_frame.paragraphs[0]
        p.runs[0].text = "Zebras orchestrate 0.67 quantum blockchain synergies across marketplaces entirely."
        R._save(prs, dst)
        return {"changed": True, "affected_slides": [8], "affected_bullets": ["s8.b1"], "bullets": []}
    res = repair(deck, "harm", repairers={"support": harmful})
    h = res["history"]
    r1 = h["rounds"][0]
    assert r1["improved"] is False and r1["became_best"] is False and r1["new_J"] > r1["previous_J"]
    assert h["final"]["selected_round"] == 0 and h["final"]["J"] == h["initial"]["J"] and not h["final"]["improved_over_initial"]
    assert (x["tmp"] / "harm" / "final.pptx").read_bytes() == deck.read_bytes()
    assert "support" in r1["blocked_after_round"] and h["stop_reason"] == "no_unblocked_repair_type_available"
    assert h["rounds_used"] == 1                                     # a blocked family is not retried


# --------------------------------------------------------------------------------- 9 min J pick #
def test_best_candidate_is_the_minimum_j_not_the_last():
    x = pipe()
    deck = tamper(t_small_font([2, 3, 4]), "best.pptx")
    calls = {"n": 0}

    def improve_then_damage(src, dst, boxes, mf):
        calls["n"] += 1
        if calls["n"] == 1:
            return R.repair_overflow(src, dst, boxes[:1], mf)        # round 1: 3 -> 2 overflowing boxes (better)
        prs = Presentation(str(src))                                 # round 2: damage (more overflow)
        for i in (5, 6, 7):
            prs.slides[i].placeholders[1].text_frame.paragraphs[0].runs[0].font.size = Pt(8)
        R._save(prs, dst)
        return {"changed": True, "affected_slides": [6, 7, 8], "boxes": []}
    res = repair(deck, "bestj", repairers={"overflow": improve_then_damage})
    h = res["history"]
    js = [h["initial"]["J"]] + [r["new_J"] for r in h["rounds"] if r["new_J"] is not None]
    assert len(h["rounds"]) == 2 and h["rounds"][0]["became_best"] and not h["rounds"][1]["became_best"]
    assert h["rounds"][1]["new_J"] > h["rounds"][0]["new_J"]
    assert h["final"]["J"] == min(js) == h["rounds"][0]["new_J"] and h["final"]["selected_round"] == 1
    assert res["verification"]["objective"] == round(min(js), 6)
    assert res["verification"]["details"]["overflow"]["overflow_count"] == 2          # round-1 deck, not the last one


# ---------------------------------------------------------------------- 10 explicit priority order #
def test_repair_priority_overflow_then_support_then_preference():
    deck = tamper(lambda prs: (t_small_font([2])(prs), t_unsupported(prs), t_number(prs), t_titles(prs)), "multi.pptx")
    b = verify(deck)
    assert b["overflow_rate"] > 0 and b["sr"] < 1 and b["nf"] < 1 and b["dalloc"] > 0
    res = repair(deck, "multi")
    h, a = res["history"], res["verification"]
    r1 = h["rounds"][0]
    order = [x["family"] for x in r1["actions"] if x["changed"]]
    assert order[:3] == ["overflow", "support", "preference"], order
    assert [f["family"] for f in r1["detected_failures"]] == sorted((f["family"] for f in r1["detected_failures"]), key=R.FAMILIES.index)
    assert a["overflow_rate"] == 0 and a["sr"] == 1 and a["nf"] == 1 and a["dalloc"] == 0 and a["dorder"] == 0
    assert a["objective"] < b["objective"] and h["final"]["improved_over_initial"]
    assert set(h["final"]["repair_types_applied"]) >= {"overflow", "support_fidelity", "preference"}


# --------------------------------------------------------------------------- 11 nothing to repair #
def test_unrepairable_failure_changes_nothing_and_stops():
    x = pipe()
    res = repair(x["deck"], "dstyle_only", cfg=R.RepairConfig(max_preference_distance=0.01))   # Dstyle (not repairable)
    h = res["history"]
    assert h["stop_reason"] == "no_effective_repair" and h["rounds_used"] == 1
    assert h["rounds"][0]["improved"] is False and h["rounds"][0]["candidate_file"] is None
    assert (x["tmp"] / "dstyle_only" / "final.pptx").read_bytes() == x["deck"].read_bytes()
    assert "Dstyle" in h["final"]["not_repairable_note"]


# -------------------------------------------------------------------------------- 12 auditability #
def test_repair_history_is_auditable():
    deck = tamper(lambda prs: (t_unsupported(prs), t_titles(prs)), "audit.pptx")
    res = repair(deck, "audit")
    h = res["history"]
    for k in ("schema", "config", "initial", "rounds", "rounds_used", "stop_reason", "final"):
        assert k in h
    r = h["rounds"][0]
    for k in ("round", "detected_failures", "actions", "previous_J", "new_J", "improved", "became_best", "candidate_file"):
        assert k in r, k
    assert r["round"] == 1 and r["previous_J"] == h["initial"]["J"] and r["improved"] and r["became_best"]
    for a in r["actions"]:
        assert a["why"] and a["source"] and "what" in a and "changed" in a and "affected_slides" in a     # WHY / SOURCE / WHAT
    sup = next(a for a in r["actions"] if a["repair_type"] == "support_fidelity")
    b = sup["what"]["bullets"][0]
    assert b["old_text"].startswith("We evaluate on seven") and b["restored_from"] and b["source_pointer"]["sentence_id"].startswith("S")
    assert sup["affected_bullets"] == ["s7.b1"] and "sr = " in sup["why"][0]
    assert h["final"]["J"] < h["initial"]["J"] and h["final"]["improved_over_initial"]       # RESULT
    assert h["config"]["max_rounds"] == 3 and h["config"]["repair_priority"] == ["overflow", "support", "preference"]


# ------------------------------------------------------------------------------ 13 final outputs #
def test_final_outputs_exist_and_open():
    x = pipe()
    out = x["tmp"] / "phase6"                                       # produced by the real CLI run in pipe()
    for f in ("final.pptx", "final_verification.json", "repair_history.json"):
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    prs = Presentation(str(out / "final.pptx"))                    # editable, openable PPTX
    assert len(prs.slides) == 12 and prs.slides[1].shapes.title.text_frame.text
    v = json.loads((out / "final_verification.json").read_text(encoding="utf-8"))
    h = json.loads((out / "repair_history.json").read_text(encoding="utf-8"))
    assert v["schema"] == V.SCHEMA and h["schema"] == R.SCHEMA and v["objective"] == h["final"]["J"]
    assert close(v["objective"], verify(out / "final.pptx")["objective"], 1e-9)           # JSON matches the deck on disk
    assert "AuditTailor - Phase 6" in x["stdout"] and "Repair rounds used" in x["stdout"]
    # a repaired run writes the same three files plus candidates
    res = repair(tamper(t_number, "num_out.pptx"), "outputs_repaired")
    d = x["tmp"] / "outputs_repaired"
    assert (d / "final.pptx").is_file() and (d / "candidates" / "round_1.pptx").is_file()
    assert len(Presentation(str(d / "final.pptx")).slides) == 12


# ----------------------------------------------------------------------------------- 14 determinism #
def test_repair_is_deterministic():
    x = pipe()
    deck = tamper(lambda prs: (t_small_font([2])(prs), t_unsupported(prs), t_number(prs), t_titles(prs)), "det.pptx")
    a, b = repair(deck, "det_a"), repair(deck, "det_b")
    assert json.dumps(a["history"], sort_keys=True) == json.dumps(b["history"], sort_keys=True)
    assert json.dumps(a["verification"]["objective"]) == json.dumps(b["verification"]["objective"])
    assert (x["tmp"] / "det_a" / "final.pptx").read_bytes() == (x["tmp"] / "det_b" / "final.pptx").read_bytes()
    ha = (x["tmp"] / "det_a" / "repair_history.json").read_text()
    assert ha == (x["tmp"] / "det_b" / "repair_history.json").read_text()


# ---------------------------------------------------------------------------- 15 no fabrication #
def test_no_unsupported_or_fabricated_content_is_introduced():
    x = pipe()
    allowed = {s["text"] for s in x["parsed"]["sentences"]}
    for sl in x["draft"]["slides"]:
        for b in sl["bullets"]:
            allowed |= {b["text"], b["original_text"]}
    deck = tamper(lambda prs: (t_blow_up(prs), t_unsupported(prs), t_number(prs), t_titles(prs)), "fab.pptx")
    res = repair(deck, "fab")
    final = V.read_deck_bullets(res["pptx"])
    assert len(final) == 20 and all(b["text"] in allowed for b in final)       # every bullet is Phase 3 / source text
    assert not any("zebra" in b["text"].lower() or "0.67" in b["text"] for b in final)
    v = res["verification"]
    assert v["sr"] == 1.0 and v["nf"] == 1.0 and v["overflow_rate"] == 0.0 and v["dalloc"] == 0.0
    sents = {s["id"]: s["text"] for s in x["parsed"]["sentences"]}
    for b in final:                                                            # pointers resolve and match the source
        assert b["sentence_id"] in sents and set(V.extract_numbers(b["text"])) <= set(V.extract_numbers(sents[b["sentence_id"]]))
    # the Phase 3 draft handed in was not modified in place
    assert x["draft"] == json.loads((x["tmp"] / "draft.json").read_text(encoding="utf-8"))


# -------------------------------------------------------------------------------- 16 CLI end-to-end #
def test_cli_pipeline_and_errors():
    x = pipe()
    h = json.loads((x["tmp"] / "phase6" / "repair_history.json").read_text(encoding="utf-8"))
    assert h["rounds_used"] == 0 and h["final"]["J"] == x["ver"]["objective"]             # clean deck: unchanged
    assert (x["tmp"] / "phase6" / "final.pptx").read_bytes() == x["deck"].read_bytes()
    # Phase 6 only, from files, on a tampered deck
    deck = tamper(t_number, "cli_num.pptx")
    out = x["tmp"] / "cli_p6"
    cmd = [sys.executable, str(MAIN), "--parsed", str(x["tmp"] / "p1" / "synthetic_paper_parsed.json"),
           "--qpp", str(x["tmp"] / "p1" / "synthetic_reference_qpp.json"), "--plan-json", str(x["tmp"] / "plan.json"),
           "--draft-json", str(x["tmp"] / "draft.json"), "--generated-pptx", str(deck), "--template", str(DECK),
           "--verify-out", str(x["tmp"] / "cli_ver.json"), "--repair", "--repair-out", str(out), "--max-repair-rounds", "2"]
    run = lambda c: subprocess.run(c, capture_output=True, text=True, env=single_thread_env(), timeout=CLI_TIMEOUT)
    p = run(cmd)
    assert p.returncode == 0, p.stderr
    hh = json.loads((out / "repair_history.json").read_text(encoding="utf-8"))
    assert hh["config"]["max_rounds"] == 2 and hh["final"]["metrics"]["nf"] == 1.0 and hh["final"]["improved_over_initial"]
    assert "Initial J(S)" in p.stdout and "Final J(S)" in p.stdout
    bad = run([sys.executable, str(MAIN), "--paper", str(PAPER), "--repair"])
    assert bad.returncode != 0 and "--repair needs" in bad.stderr


# ------------------------------------------------------------------------------ 17-19 earlier phases #
def test_earlier_phases_still_pass():
    from suite_runner import nested_run_by_parent
    if nested_run_by_parent():        # test_app is already running the Phase 1-6 suites itself, once each
        print("  (Phase 1-5 suites are being run by the parent suite; not repeated here)")
        return
    run_suites((1, 2, 3, 4, 5))       # standalone: Phase 1-5 suites run concurrently, each with a timeout


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import time
    for t in tests:
        t0 = time.time()
        t()
        print(f"  ok  {t.__name__}  ({time.time() - t0:.1f}s)")
    print(f"ALL PHASE 6 TESTS PASSED ({len(tests)} tests)")

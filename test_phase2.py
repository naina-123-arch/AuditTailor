"""
Phase 2 tests (budgeted planner).   Run:   python tests/test_phase2.py
(also collectable by pytest).  Standard library only, plus the Phase 1 outputs for
the end-to-end CLI test.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from planner import (SECTIONS, blend, largest_remainder, order_sections,   # noqa: E402
                     plan_slides, validate_lambda, validate_n_slides)

FIVE = ["Motivation", "Background", "Method", "Results", "Conclusion"]       # no Setup
PI_FIVE = list(FIVE)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def l1(a, b):
    return sum(abs(float(a[k]) - float(b[k])) for k in SECTIONS)


# --- the worked example from the task description (values live in the TEST only) ----------- #
EX_WORDS = {"Motivation": 15, "Background": 15, "Method": 35, "Results": 25, "Conclusion": 10}   # p = .15 .15 .35 .25 .10
EX_W = {"Motivation": .10, "Background": .20, "Method": .40, "Results": .20, "Conclusion": .10}  # w' (already sums to 1)


# 1 ---------------------------------------------------------------------------------------- #
def test_lambda_zero_follows_paper_only():
    plan = plan_slides(EX_WORDS, EX_W, PI_FIVE, 10, 0.0)
    for k in SECTIONS:
        assert plan.q[k] == plan.p[k]                          # q = p exactly
    # N*p = 1.5 1.5 3.5 2.5 1.0 -> floors 1 1 3 2 1 (=8), m=2, remainders .5 .5 .5 .5 0
    # ties -> larger q: Method (.35) and Results (.25)
    assert plan.allocation == {"Motivation": 1, "Background": 1, "Method": 4, "Setup": 0, "Results": 3, "Conclusion": 1}


# 2 ---------------------------------------------------------------------------------------- #
def test_lambda_one_follows_presenter_only():
    plan = plan_slides(EX_WORDS, EX_W, PI_FIVE, 10, 1.0)
    for k in SECTIONS:
        assert plan.q[k] == plan.w_prime[k]                    # q = w' exactly
    assert plan.allocation == {"Motivation": 1, "Background": 2, "Method": 4, "Setup": 0, "Results": 2, "Conclusion": 1}


# 3 ---------------------------------------------------------------------------------------- #
def test_paper_missing_sections_renormalises_and_assigns_nothing():
    words = {"Motivation": 100, "Method": 300, "Results": 100}                # Background, Setup, Conclusion absent
    w = {"Motivation": .1, "Background": .2, "Method": .3, "Setup": .1, "Results": .2, "Conclusion": .1}
    plan = plan_slides(words, w, SECTIONS, 8, 0.5)
    assert plan.present == ["Motivation", "Method", "Results"]
    assert plan.absent == ["Background", "Setup", "Conclusion"]
    assert sum(plan.w_prime.values()) == 1                                    # renormalised: .1/.6 .3/.6 .2/.6
    assert plan.w_prime["Method"] == Fraction(1, 2)
    assert all(plan.allocation[k] == 0 for k in plan.absent)
    assert sum(plan.allocation.values()) == 8
    assert plan.order == ["Motivation", "Method", "Results"]                  # absent sections skipped
    assert any("Background" in w_ for w_ in plan.warnings)                    # dropped reference weight is reported


def test_zero_reference_weight_on_present_sections_falls_back_to_p():
    words = {"Method": 200, "Results": 200}
    w = {"Motivation": .5, "Conclusion": .5}                                  # reference only cares about absent sections
    plan = plan_slides(words, w, SECTIONS, 4, 1.0)
    assert plan.w_prime["Method"] == plan.p["Method"] == Fraction(1, 2)
    assert plan.fallbacks and plan.allocation["Method"] == 2 and plan.allocation["Results"] == 2


# 4 ---------------------------------------------------------------------------------------- #
def test_n_not_divisible_among_sections():
    words = {k: 100 for k in SECTIONS}                                        # equal thirds-ish: 6 sections
    w = {k: 1 / 6 for k in SECTIONS}
    for n in (7, 10, 11, 13):
        plan = plan_slides(words, w, SECTIONS, n, 0.7)
        assert sum(plan.allocation.values()) == n
        assert max(plan.allocation.values()) - min(plan.allocation.values()) <= 1   # even split, remainder spread


# 5 ---------------------------------------------------------------------------------------- #
def test_largest_remainder_user_example():
    plan = plan_slides(EX_WORDS, EX_W, PI_FIVE, 10, 0.6)
    expect_q = {"Motivation": .12, "Background": .18, "Method": .38, "Setup": 0, "Results": .22, "Conclusion": .10}
    for k, v in expect_q.items():
        assert abs(float(plan.q[k]) - v) < 1e-12
    d = plan.detail
    assert [d[k]["floor"] for k in FIVE] == [1, 1, 3, 2, 1]                   # sum 8 -> m = 2
    assert d["Background"]["extra"] and d["Method"]["extra"]                  # remainders .8 and .8 are the largest
    assert plan.allocation == {"Motivation": 1, "Background": 2, "Method": 4, "Setup": 0, "Results": 2, "Conclusion": 1}


def test_largest_remainder_matches_proposal_table2_toy_example():
    # Proposal Table 2 (Experiments -> Results): ck = (2,3,5,2), words (1200,2400,1800,600), lambda .6, N=10
    words = {"Background": 1200, "Method": 2400, "Results": 1800, "Conclusion": 600}
    w = {"Background": Fraction(2, 12), "Method": Fraction(3, 12), "Results": Fraction(5, 12), "Conclusion": Fraction(2, 12)}
    plan = plan_slides(words, w, ["Background", "Method", "Results", "Conclusion"], 10, 0.6)
    assert [plan.detail[k]["floor"] for k in ("Background", "Method", "Results", "Conclusion")] == [1, 3, 3, 1]
    assert plan.detail["Background"]["extra"] and plan.detail["Results"]["extra"]      # remainders .8 and .7
    assert [plan.allocation[k] for k in ("Background", "Method", "Results", "Conclusion")] == [2, 3, 4, 1]


def test_largest_remainder_tie_break_prefers_larger_q_then_canonical_order():
    # N=2, q = (.25,.25,.5): raw .5 .5 1.0 -> floors 0 0 1, m=1, tied remainders -> canonical order picks Motivation
    q = {k: Fraction(0) for k in SECTIONS}
    q.update({"Motivation": Fraction(1, 4), "Background": Fraction(1, 4), "Method": Fraction(1, 2)})
    d = largest_remainder(q, 2, ["Motivation", "Background", "Method"])
    assert (d["Motivation"]["slides"], d["Background"]["slides"], d["Method"]["slides"]) == (1, 0, 1)
    # N=3, q = (.3,.2,.5): raw .9 .6 1.5 -> floors 0 0 1, m=2 -> remainders .9 .6 .5 -> Motivation, Background
    q = {k: Fraction(0) for k in SECTIONS}
    q.update({"Motivation": Fraction(3, 10), "Background": Fraction(2, 10), "Method": Fraction(5, 10)})
    d = largest_remainder(q, 3, ["Motivation", "Background", "Method"])
    assert (d["Motivation"]["slides"], d["Background"]["slides"], d["Method"]["slides"]) == (1, 1, 1)
    # equal remainders, different q: N=3, q=(.4, .2333.., .3666..)? use exact: q=(1/3+1/30, 1/3-1/30... ) simple case below
    q = {k: Fraction(0) for k in SECTIONS}
    q.update({"Motivation": Fraction(1, 6), "Background": Fraction(1, 2), "Method": Fraction(1, 3)})
    d = largest_remainder(q, 3, ["Motivation", "Background", "Method"])     # raw .5 1.5 1.0 -> m=1, tie .5/.5 -> larger q Background
    assert d["Background"]["extra"] and not d["Motivation"]["extra"]


def test_float_noise_does_not_break_ties():
    # 10 * 0.18 is 1.7999999999999998 in floats; exact arithmetic must still see remainder 0.8 == 0.8
    plan = plan_slides(EX_WORDS, EX_W, PI_FIVE, 10, 0.6)
    assert plan.detail["Background"]["remainder"] == Fraction(4, 5)
    assert plan.detail["Method"]["remainder"] == Fraction(4, 5)


# 6 + 7 ------------------------------------------------------------------------------------ #
def test_total_always_equals_n_and_absent_sections_get_nothing_randomised():
    rng = random.Random(20260110)
    for _ in range(1500):
        present = [k for k in SECTIONS if rng.random() < 0.7] or [rng.choice(SECTIONS)]
        words = {k: (rng.randint(1, 3000) if k in present else 0) for k in SECTIONS}
        w = {k: rng.random() * (rng.random() < 0.8) for k in SECTIONS}
        pi = rng.sample(SECTIONS, rng.randint(0, 6))
        n = rng.randint(1, 40)
        lam = rng.choice([0, 1, 0.5, round(rng.random(), 3)])
        plan = plan_slides(words, w, pi, n, lam)
        assert sum(plan.allocation.values()) == n
        assert all(plan.allocation[k] >= 0 for k in SECTIONS)
        for k in SECTIONS:
            if words[k] == 0:
                assert plan.allocation[k] == 0 and plan.q[k] == 0
            else:
                assert abs(plan.allocation[k] - n * plan.q[k]) < 1             # proposal's rounding bound
        assert sum(plan.q.values()) == 1 and sum(plan.w_prime.values()) == 1
        assert sorted(plan.order) == sorted(plan.present)                      # every present section appears once


def test_blend_is_closer_to_presenter_than_paper_for_positive_lambda():
    # success criterion in Sec. 4.3, checked on the continuous blend (integer rounding can blur it)
    words = {"Motivation": 100, "Background": 300, "Method": 200, "Results": 300, "Conclusion": 100}
    w = {"Motivation": .3, "Background": .05, "Method": .45, "Results": .1, "Conclusion": .1}
    for lam in (0.1, 0.5, 0.9):
        plan = plan_slides(words, w, FIVE, 10, lam)
        assert l1(plan.q, plan.w_prime) < l1(plan.p, plan.w_prime)


# 8 ---------------------------------------------------------------------------------------- #
def test_section_order_follows_pi():
    pi = ["Results", "Method", "Motivation", "Conclusion", "Background", "Setup"]
    plan = plan_slides({k: 50 for k in SECTIONS}, {k: 1 for k in SECTIONS}, pi, 12, 0.5)
    assert plan.order == pi and plan.inserted == []
    pi2 = ["Method", "Setup", "Results", "Motivation", "Conclusion", "Background"]
    assert plan_slides({k: 50 for k in SECTIONS}, {k: 1 for k in SECTIONS}, pi2, 12, 0.5).order == pi2


# 9 ---------------------------------------------------------------------------------------- #
def test_missing_reference_section_inserted_before_conclusion():
    pi = ["Motivation", "Method", "Results", "Conclusion"]                    # Background and Setup missing from pi
    order, inserted = order_sections(pi, SECTIONS)
    assert order == ["Motivation", "Method", "Results", "Background", "Setup", "Conclusion"]
    assert inserted == ["Background", "Setup"]
    plan = plan_slides({k: 50 for k in SECTIONS}, {k: 1 for k in SECTIONS}, pi, 12, 0.5)
    assert plan.order == order and plan.inserted == ["Background", "Setup"]
    # Conclusion not in the paper -> missing sections go to the end
    assert order_sections(["Method"], ["Method", "Results"])[0] == ["Method", "Results"]
    # Conclusion itself missing from pi: other missing sections first, Conclusion last
    assert order_sections(["Method"], ["Motivation", "Method", "Conclusion"])[0] == ["Method", "Motivation", "Conclusion"]
    # empty pi (reference had no content slides): canonical order, Conclusion last, with a warning
    plan = plan_slides({k: 50 for k in SECTIONS}, {k: 0 for k in SECTIONS}, [], 6, 0.5)
    assert plan.order == SECTIONS and plan.fallbacks and any("pi is empty" in w_ for w_ in plan.warnings)


# 10 --------------------------------------------------------------------------------------- #
def test_invalid_lambda_rejected():
    for bad in (-0.1, 1.0000001, 2, -1, float("nan"), float("inf"), float("-inf"), "0.5", None, True, [0.5]):
        assert raises(ValueError, validate_lambda, bad), bad
        assert raises(ValueError, plan_slides, EX_WORDS, EX_W, PI_FIVE, 10, bad), bad
    for ok in (0, 1, 0.0, 1.0, 0.6, Fraction(1, 3)):
        validate_lambda(ok)


def test_invalid_n_and_inputs_rejected():
    for bad in (0, -3, 10.5, 10.0, "10", None, True):
        assert raises(ValueError, validate_n_slides, bad), bad
    assert raises(ValueError, plan_slides, {k: 0 for k in SECTIONS}, EX_W, PI_FIVE, 10, 0.5)       # empty paper
    assert raises(ValueError, plan_slides, {"Nonsense": 10}, EX_W, PI_FIVE, 10, 0.5)               # unknown label
    assert raises(ValueError, plan_slides, EX_WORDS, {"Method": -0.2}, PI_FIVE, 10, 0.5)           # negative weight
    assert raises(ValueError, plan_slides, EX_WORDS, EX_W, ["Method", "Bogus"], 10, 0.5)           # unknown label in pi
    assert raises(ValueError, plan_slides, {"Method": 10.5}, EX_W, PI_FIVE, 10, 0.5)               # fractional word count


def test_small_n_leaves_sections_unpresented_with_warning():
    plan = plan_slides({k: 100 for k in SECTIONS}, {k: 1 for k in SECTIONS}, SECTIONS, 3, 0.5)
    assert sum(plan.allocation.values()) == 3
    assert len(plan.presented_order) == 3 and sum(1 for k in plan.present if plan.allocation[k] == 0) == 3
    assert any("smaller than the number of present sections" in w_ for w_ in plan.warnings)


# end-to-end through the CLI (Phase 1 -> Phase 2) ------------------------------------------- #
def test_cli_end_to_end_plan_json_and_console():
    from make_synthetic_data import make_pdf, make_pptx
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf, pptx = tmp / "paper.pdf", tmp / "deck.pptx"
        make_pdf(pdf)
        make_pptx(pptx)
        plan_path = tmp / "p2" / "plan.json"
        base = [sys.executable, str(ROOT / "src" / "main.py")]

        proc = subprocess.run(base + ["--paper", str(pdf), "--deck", str(pptx), "--out", str(tmp / "p1"),
                                      "--plan", "--n-slides", "10", "--lambda", "0.6", "--plan-out", str(plan_path)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        for needle in ("AuditTailor - Phase 2", "Budgeted Slide Planner", "Total slides: 10", "Lambda: 0.60",
                       "PAPER DISTRIBUTION", "REFERENCE DISTRIBUTION", "BLENDED DISTRIBUTION",
                       "FINAL SLIDE ALLOCATION", "Total         : 10"):
            assert needle in proc.stdout, f"console output missing {needle!r}"

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for key in ("total_slides", "lambda", "paper_distribution", "reference_distribution",
                    "blended_distribution", "slide_allocation", "section_order"):
            assert key in plan, key
        assert plan["total_slides"] == 10 and plan["lambda"] == 0.6
        assert sum(plan["slide_allocation"].values()) == 10
        qpp = json.loads((tmp / "p1" / "deck_qpp.json").read_text(encoding="utf-8"))
        assert plan["section_order"] == qpp["theta"]["pi"]                       # all six present -> order == pi
        for k in SECTIONS:                                                      # |n - N q| < 1 using the JSON values
            assert abs(plan["slide_allocation"][k] - 10 * plan["blended_distribution"][k]) < 1.0 + 1e-5
        # recompute q from the Phase 1 JSON values as an independent check of Eq. (3)
        for k in SECTIONS:
            q = 0.6 * plan["reference_distribution_restricted"][k] + 0.4 * plan["paper_distribution"][k]
            assert abs(q - plan["blended_distribution"][k]) < 2e-6

        # planning from the Phase 1 JSON files only (no PDF/PPTX parsing)
        plan2 = tmp / "p2" / "plan_from_json.json"
        proc = subprocess.run(base + ["--parsed", str(tmp / "p1" / "paper_parsed.json"), "--qpp", str(tmp / "p1" / "deck_qpp.json"),
                                      "--plan", "--n-slides", "10", "--lambda", "0.6", "--plan-out", str(plan2)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(plan2.read_text())["slide_allocation"] == plan["slide_allocation"]

        # invalid lambda is rejected by the CLI with a non-zero exit code and a clear message
        proc = subprocess.run(base + ["--parsed", str(tmp / "p1" / "paper_parsed.json"), "--qpp", str(tmp / "p1" / "deck_qpp.json"),
                                      "--plan", "--n-slides", "10", "--lambda", "1.5", "--plan-out", str(tmp / "x.json")],
                              capture_output=True, text=True)
        assert proc.returncode == 1 and "lambda must be in [0, 1]" in proc.stderr and not (tmp / "x.json").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL PHASE 2 TESTS PASSED ({len(tests)} tests)")

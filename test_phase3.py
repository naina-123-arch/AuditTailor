"""
Phase 3 tests (key-free extractive drafting: TF-IDF + MMR).   Run:   python tests/test_phase3.py
(also collectable by pytest).  Uses synthetic papers; the CLI test runs Phase 1 -> 2 -> 3 end to end.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from extractor import (DEFAULT_BULLETS_PER_SLIDE, SECTIONS, bullets_per_slide, build_tfidf,   # noqa: E402
                       cosine_matrix, draft_deck, evidence_indicator, load_candidates, mmr_select,
                       relevance_scores, split_even, trim_sentence, validate_weights)
from planner import plan_slides                                                          # noqa: E402


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- synthetic fixtures #
def sent(sec: str, i: int) -> str:
    return f"The {sec.lower()} stage number{i} evaluates unique{sec.lower()}{i} component behaviour carefully."


def make_parsed(counts=None, title="Learned Sorting of Widgets",
                abstract="We propose a learned sorter for widgets. Each stage evaluates component behaviour carefully.", extra=None):
    """Phase 1-style parsed dict: ``counts`` = number of sentences per canonical section."""
    counts = counts or {k: 10 for k in SECTIONS}
    sentences, i = [], 0
    for sec in SECTIONS:
        texts = [sent(sec, j) for j in range(counts.get(sec, 0))] + list((extra or {}).get(sec, []))
        for t in texts:
            sentences.append({"id": i, "page": 1 + i // 5, "section_index": SECTIONS.index(sec) + 1,
                              "section": f"{SECTIONS.index(sec) + 1} {sec} heading", "label": sec, "text": t})
            i += 1
    return {"title": title, "abstract": abstract, "sentences": sentences, "captions": []}


def make_plan(alloc, q=None):
    full = {k: alloc.get(k, 0) for k in SECTIONS}
    order = [k for k in SECTIONS if full[k] > 0]
    tot = sum(full.values())
    q = q or {k: full[k] / tot for k in SECTIONS}
    return {"total_slides": tot, "slide_allocation": full, "section_order": order,
            "presented_section_order": order, "blended_distribution": q}


QPP2 = {"mean_bullets": 2.0}
ALLOC = {"Motivation": 1, "Background": 1, "Method": 2, "Setup": 1, "Results": 3, "Conclusion": 1}   # N = 9


def by_id(parsed):
    return {f"S{s['id']:03d}": s for s in parsed["sentences"]}


def words(t: str):
    return re.findall(r"[A-Za-z0-9]+", t.lower())


# ------------------------------------------------------------------------------------ 1. TF-IDF #
def test_tfidf_vectors_generated():
    texts = ["The learned sorter predicts bucket boundaries with a neural network.",
             "Accuracy rises from 0.76 to 0.87 on the benchmark suite.",
             "Baselines are quicksort, mergesort and a learned index."]
    sp = build_tfidf(texts, "Learned sorting of widgets. A neural sorter for buckets.")
    assert sp.raw.shape[0] == 3 and sp.unit.shape == sp.raw.shape and sp.n_terms > 5
    norms = np.sqrt(np.asarray(sp.unit.multiply(sp.unit).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0)                                         # L2-normalised rows
    vocab = set(sp.vectorizer.vocabulary_)
    assert "the" not in vocab and not any(re.fullmatch(r"[\d.]+", t) for t in vocab)   # no stop words / bare numbers
    assert sp.query_unit is not None
    sim = cosine_matrix(sp.unit)
    assert np.allclose(np.diag(sim), 1.0) and np.allclose(sim, sim.T)
    assert raises(ValueError, build_tfidf, [])


# ------------------------------------------------------------------------------- 2. relevance #
def test_evidence_indicator():
    yes = ["Accuracy rises from 0.76 to 0.87 on the benchmark.", "Runtime drops by 31 percent on sorted inputs.",
           "Error falls to 4% on the test split.", "Set N = 10 for all of the runs.", "Latency was 12 ms per query."]
    no = ["The model is trained on a large corpus of text.", "We discuss Section 3 and Table 2 in detail.",
          "As reported in [5], the approach is popular.", "Since 2021 the method improves with scale."]
    assert all(evidence_indicator(t) == 1 for t in yes), [t for t in yes if evidence_indicator(t) != 1]
    assert all(evidence_indicator(t) == 0 for t in no), [t for t in no if evidence_indicator(t) != 0]


def test_relevance_scores_generated():
    parsed = make_parsed({"Motivation": 3, "Method": 3, "Results": 3},
                         extra={"Results": ["Accuracy rises from 0.76 to 0.87 for the learned sorter of widgets."]})
    cands, _ = load_candidates(parsed)
    sp = build_tfidf([c.text for c in cands], parsed["title"] + " " + parsed["abstract"])
    q = {"Motivation": 0.2, "Method": 0.5, "Results": 0.3}
    comp = relevance_scores(sp, cands, q)
    n = len(cands)
    for key in ("rel", "topical", "keyword", "section", "evidence"):
        assert comp[key].shape == (n,) and np.all(comp[key] >= 0) and np.all(comp[key] <= 1 + 1e-9), key
    for i, c in enumerate(cands):                                          # section signal = q_k / max q
        assert abs(comp["section"][i] - q[c.section] / 0.5) < 1e-9
    ev = [i for i, c in enumerate(cands) if "rises" in c.text][0]
    assert comp["evidence"][ev] == 1 and comp["topical"][ev] > comp["topical"][0]   # shares 'learned sorter widgets' with the title
    w = validate_weights((1, 0, 0, 0))                                      # one-hot weights -> Rel equals that signal
    assert np.allclose(relevance_scores(sp, cands, q, w)["rel"], comp["topical"])
    assert np.allclose(validate_weights((1, 1, 1, 1)), 0.25)
    for bad in ((1, 2, 3), (-1, 1, 1, 1), (0, 0, 0, 0), (float("nan"), 1, 1, 1)):
        assert raises(ValueError, validate_weights, bad)


# ------------------------------------------------------------------------------------ 3. MMR #
S1 = "The learned sorter predicts bucket boundaries with a small neural network."
S2 = "The learned sorter predicts bucket boundaries using a small neural network model."
S3 = "Experiments cover three synthetic datasets with one million widgets each."


def _mmr_fixture():
    sp = build_tfidf([S1, S2, S3, "Quicksort and mergesort are classical comparison baselines."])
    return cosine_matrix(sp.unit)[:3, :3], np.array([0.90, 0.85, 0.60])


def test_mmr_selects_multiple_and_first_is_top_relevance():
    sim, rel = _mmr_fixture()
    picks = mmr_select(rel, sim, 3, 0.7)
    assert [p.pos for p in picks][0] == 0 and len(picks) == 3 and len({p.pos for p in picks}) == 3
    assert picks[0].max_sim == 0.0 and [p.rank for p in picks] == [1, 2, 3]
    rng = np.random.RandomState(0)                                          # first pick = argmax Rel for any lambda_m > 0
    for _ in range(20):
        r = rng.rand(8)
        s = np.clip(rng.rand(8, 8), 0, 1)
        s = (s + s.T) / 2
        np.fill_diagonal(s, 1)
        for lam in (0.05, 0.5, 1.0):
            assert mmr_select(r, s, 3, lam)[0].pos == int(np.argmax(r))
    assert len(mmr_select(rel, sim, 10, 0.7)) == 3                          # asking for more than the pool returns the pool


def test_mmr_reduces_obvious_redundancy():
    sim, rel = _mmr_fixture()
    assert sim[0, 1] > 0.6 and sim[0, 2] < 0.2                              # S2 is a near-duplicate of S1, S3 is different
    mmr = [p.pos for p in mmr_select(rel, sim, 2, 0.7)]
    assert mmr == [0, 2]                                                    # S1, then S3 (not the near-duplicate S2)
    pure = [p.pos for p in mmr_select(rel, sim, 2, 1.0)]
    assert pure == [0, 1]                                                   # lambda_m = 1 is plain relevance ranking
    pen = mmr_select(rel, sim, 2, 0.7)[1]
    assert abs(pen.mmr - (0.7 * 0.60 - 0.3 * sim[0, 2])) < 1e-9             # Eq. (4) value of the 2nd pick
    assert [p.pos for p in mmr_select(rel, sim, 2, 0.0)] == [0, 2]          # lambda_m = 0 -> pure diversity after the first pick
    assert raises(ValueError, mmr_select, rel, sim, 2, 1.2)


def test_mmr_tie_break_is_deterministic():
    rel = np.array([0.5, 0.5, 0.5])
    sim = np.eye(3)
    assert [p.pos for p in mmr_select(rel, sim, 3, 0.7, tie_keys=[30, 10, 20])] == [1, 2, 0]


# ------------------------------------------------------------------------------- 4. the drafter #
def test_selected_sentences_come_from_source_and_are_not_duplicated():
    parsed = make_parsed()
    d = draft_deck(parsed, QPP2, make_plan(ALLOC))
    src = by_id(parsed)
    seen = []
    for s in d["slides"]:
        for b in s["bullets"]:
            sid = b["source"]["sentence_id"]
            assert sid in src
            assert b["original_text"] == src[sid]["text"]                   # untouched source sentence
            assert set(words(b["text"])) <= set(words(b["original_text"]))  # trimmed text adds no words
            seen.append(sid)
    assert seen and len(seen) == len(set(seen))                             # no sentence twice


def test_section_order_slide_count_and_budget_follow_plan():
    parsed = make_parsed()
    words_k = {k: 100 for k in SECTIONS}
    plan = plan_slides(words_k, {"Motivation": .1, "Background": .1, "Method": .3, "Setup": .1, "Results": .3, "Conclusion": .1},
                       SECTIONS, 10, 0.6).to_dict()
    d = draft_deck(parsed, QPP2, plan)
    assert d["n_slides"] == len(d["slides"]) == plan["total_slides"] == 10
    assert [s["slide_number"] for s in d["slides"]] == list(range(1, 11))
    expected = [k for k in plan["presented_section_order"] for _ in range(plan["slide_allocation"][k])]
    assert [s["section"] for s in d["slides"]] == expected                  # order and counts exactly as Phase 2
    b_bar = d["parameters"]["bullets_per_slide_b_bar"]
    assert b_bar == 2 and d["warnings"] == []
    for s in d["slides"]:
        assert len(s["bullets"]) == b_bar                                   # n_k * b_bar spread evenly over n_k slides
        assert all(b["source"]["section"] == s["section"] for b in s["bullets"])
    for k, v in d["section_summary"].items():
        assert v["selected"] == v["requested_bullets"] == plan["slide_allocation"][k] * b_bar


def test_bullets_sit_in_paper_order_inside_a_section():
    d = draft_deck(make_parsed(), QPP2, make_plan(ALLOC))
    for k in SECTIONS:
        ids = [int(b["source"]["sentence_id"][1:]) for s in d["slides"] if s["section"] == k for b in s["bullets"]]
        assert ids == sorted(ids)


def test_source_pointers_preserved():
    parsed = make_parsed()
    d = draft_deck(parsed, QPP2, make_plan(ALLOC))
    src = by_id(parsed)
    for s in d["slides"]:
        for b in s["bullets"]:
            p = b["source"]
            assert re.fullmatch(r"S\d{3}", p["sentence_id"])
            assert p["page"] == src[p["sentence_id"]]["page"] and p["section"] == src[p["sentence_id"]]["label"]
            assert p["heading"] == src[p["sentence_id"]]["section"]
            assert set(b["selection"]) >= {"mmr_rank", "relevance", "max_similarity_to_selected", "mmr_score"}


def test_insufficient_candidates_warn_and_nothing_is_invented():
    parsed = make_parsed({"Motivation": 4, "Background": 4, "Method": 8, "Setup": 4, "Results": 2, "Conclusion": 4})
    d = draft_deck(parsed, QPP2, make_plan(ALLOC), handover=False)         # Results: 3 slides x 2 = 6 requested, 2 exist
    assert "WARNING: Results section requested 6 bullets but only 2 suitable candidates were available." in d["warnings"]
    res = [b for s in d["slides"] if s["section"] == "Results" for b in s["bullets"]]
    assert len(res) == 2 and d["section_summary"]["Results"]["selected"] == 2
    assert d["n_slides"] == 9                                               # allocation unchanged
    assert any("has no bullets" in w for w in d["warnings"])               # an empty slide stays empty
    src = {s["text"] for s in parsed["sentences"]}
    assert all(b["original_text"] in src for s in d["slides"] for b in s["bullets"])


def test_quota_handover_follows_the_proposal():
    parsed = make_parsed({"Motivation": 4, "Background": 4, "Method": 8, "Setup": 4, "Results": 2, "Conclusion": 4})
    q = {"Motivation": .1, "Background": .1, "Method": .3, "Setup": .1, "Results": .3, "Conclusion": .1}
    d = draft_deck(parsed, QPP2, make_plan(ALLOC, q))                      # default handover=True
    assert any(w.startswith("WARNING: Results section requested 6 bullets but only 2") for w in d["warnings"])
    assert d["n_slides"] == 9 and d["section_summary"]["Results"]["given_via_handover"] == 4
    assert d["section_summary"]["Method"]["received_via_handover"] == 4    # next-highest q_k (Method 0.3) gets the quota first
    moved = [b for s in d["slides"] for b in s["bullets"] if b["selection"]["from_handover_of"] == "Results"]
    assert len(moved) == 4 and all(b["source"]["section"] == "Method" for b in moved)
    ids = [b["source"]["sentence_id"] for s in d["slides"] for b in s["bullets"]]
    assert len(ids) == len(set(ids)) == 18                                 # full budget met, no duplicates
    # nothing left to hand over to -> explicit warning, not invented text
    tiny = make_parsed({k: 1 for k in SECTIONS})
    d2 = draft_deck(tiny, QPP2, make_plan(ALLOC))
    assert any("could not be reassigned" in w for w in d2["warnings"]) and d2["n_bullets"] == 6


def test_deterministic_repeated_runs():
    parsed = make_parsed()
    runs = [json.dumps(draft_deck(parsed, QPP2, make_plan(ALLOC)), sort_keys=True) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_candidate_filter_excludes_non_content():
    extra = {"Method": ["Figure 3: Overview of the learned sorter and its encoder.",
                        "Too short here.",
                        "and then the rest of the sentence continues without a start.",
                        "Hidden fragment without any final punctuation in the text",
                        "Let P = {k : l_k > 0} and w = w_k / sum over j in P of w_j here.",
                        "(2) A preference-distance score and a verify-and-repair loop.",
                        "We propose a learned sorter for widgets."]}              # = an abstract sentence
    parsed = make_parsed({"Method": 3}, extra=extra)
    parsed["sentences"].append({"id": 99, "page": 1, "section_index": 0, "section": "Abstract", "label": None,
                                "text": "An unlabelled sentence that is long enough to be a candidate."})
    cands, excl = load_candidates(parsed)
    assert len(cands) == 3
    for reason in ("caption", "too_short", "fragment_start", "fragment_end", "math_heavy", "list_item_fragment",
                   "abstract", "no_canonical_label"):
        assert excl.get(reason) == 1, (reason, excl)


# ------------------------------------------------------------------------------------ trimming #
def test_trim_is_conservative():
    t, ops = trim_sentence("Maheshwari et al. propose a graph-based method [3], which we compare against [4, 5].")
    assert t == "Maheshwari et al. propose a graph-based method, which we compare against." and ops == ["remove_citation_brackets"]
    t, ops = trim_sentence("This follows the approach shown in [3].")                  # would leave 'in .' -> keep the bracket
    assert t == "This follows the approach shown in [3]." and ops == []
    t, ops = trim_sentence("Models are compared on accuracy (Zeng et al., 2025) across all of the datasets.")
    assert t == "Models are compared on accuracy across all of the datasets." and "remove_parenthetical" in ops
    keep = "The accuracy rose to 0.87 (from 0.76 on the baseline) in our final experiment."
    assert trim_sentence(keep) == (keep, [])                                           # digits inside the parenthetical -> keep
    keep2 = "Quicksort (e.g., Hoare) is a classical baseline for sorting widgets quickly."
    assert trim_sentence(keep2) == (keep2, [])                                         # short parenthetical -> keep
    long = ("The learned sorter first computes an embedding of every widget with a small encoder, and then buckets "
            "the embeddings with a neural predictor that was trained on synthetic data of one million widgets each.")
    t, ops = trim_sentence(long, max_words=25)
    assert ops == ["clause_cut"]
    assert t == "The learned sorter first computes an embedding of every widget with a small encoder."
    assert long.startswith(t[:-1])                                                     # a true prefix of the source
    unsplittable = " ".join(["word"] * 50) + "."
    assert trim_sentence(unsplittable, max_words=35) == (unsplittable, [])             # no safe boundary -> untouched
    short = "A short sentence with six words."
    assert trim_sentence(short) == (short, [])


def test_helpers():
    assert bullets_per_slide(None)[0] == DEFAULT_BULLETS_PER_SLIDE == 4
    assert bullets_per_slide(0)[0] == 4 and bullets_per_slide(float("nan"))[0] == 4
    assert bullets_per_slide(4.2)[0] == 4 and bullets_per_slide(2.5)[0] == 3 and bullets_per_slide(1.78)[0] == 2
    assert bullets_per_slide(0.3)[0] == 1
    assert split_even(list(range(7)), 3) == [[0, 1, 2], [3, 4], [5, 6]] and split_even([], 2) == [[], []]
    d = draft_deck(make_parsed({k: 20 for k in SECTIONS}), {}, make_plan(ALLOC))      # no reference bullet count -> 4 per slide
    assert d["parameters"]["bullets_per_slide_b_bar"] == 4
    assert all(len(s["bullets"]) == 4 for s in d["slides"])


def test_invalid_inputs_are_rejected():
    parsed, plan = make_parsed(), make_plan(ALLOC)
    assert raises(ValueError, draft_deck, parsed, QPP2, plan, lambda_m=1.5)
    assert raises(ValueError, draft_deck, parsed, QPP2, plan, lambda_m=float("nan"))
    assert raises(ValueError, draft_deck, parsed, QPP2, plan, rel_weights=(1, 1, 1))
    bad = dict(plan, total_slides=99)
    assert raises(ValueError, draft_deck, parsed, QPP2, bad)
    assert raises(ValueError, draft_deck, parsed, QPP2, {"total_slides": 3})
    assert raises(ValueError, draft_deck, {"sentences": []}, QPP2, plan)


def test_missing_title_abstract_is_reported_not_hidden():
    d = draft_deck(make_parsed(title="", abstract=""), QPP2, make_plan(ALLOC))
    assert any("no title/abstract" in w for w in d["warnings"]) and d["n_bullets"] == 18


def test_extractor_uses_no_llm_or_network_libraries():
    tree = ast.parse((ROOT / "src" / "extractor.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"__future__", "math", "re", "dataclasses", "typing", "numpy", "sklearn", "section_tagger"}, mods


# ------------------------------------------------------------- CLI: Phase 1 -> Phase 2 -> Phase 3 #
def test_cli_end_to_end_phase1_2_3():
    from make_synthetic_data import make_pdf, make_pptx
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf, pptx = tmp / "paper.pdf", tmp / "deck.pptx"
        make_pdf(pdf)
        make_pptx(pptx)
        base = [sys.executable, str(ROOT / "src" / "main.py")]
        plan_p, draft_p = tmp / "p2" / "plan.json", tmp / "p3" / "draft.json"

        proc = subprocess.run(base + ["--paper", str(pdf), "--deck", str(pptx), "--out", str(tmp / "p1"),
                                      "--plan", "--n-slides", "10", "--lambda", "0.6", "--plan-out", str(plan_p),
                                      "--draft", "--draft-out", str(draft_p)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        for needle in ("AuditTailor - Phase 3", "Key-Free Extractive Drafting", "Slides planned: 10", "Lambda_MMR: 0.70",
                       "Mean bullets/slide:", "SLIDE 1", "Section: Motivation", "   Source: page", "Draft written to:"):
            assert needle in proc.stdout, f"console output missing {needle!r}"

        plan = json.loads(plan_p.read_text(encoding="utf-8"))
        draft = json.loads(draft_p.read_text(encoding="utf-8"))
        parsed = json.loads((tmp / "p1" / "paper_parsed.json").read_text(encoding="utf-8"))
        assert draft["schema"] == "audittailor.draft.v1" and draft["n_slides"] == plan["total_slides"] == 10
        expected = [k for k in plan["presented_section_order"] for _ in range(plan["slide_allocation"][k])]
        assert [s["section"] for s in draft["slides"]] == expected
        src = {f"S{s['id']:03d}": s for s in parsed["sentences"]}
        abstract = parsed["abstract"].lower()
        for s in draft["slides"]:
            for b in s["bullets"]:
                p = b["source"]
                assert b["original_text"] == src[p["sentence_id"]]["text"] and p["page"] == src[p["sentence_id"]]["page"]
                assert b["original_text"].lower() not in abstract          # abstract sentences are not candidates
        ids = [b["source"]["sentence_id"] for s in draft["slides"] for b in s["bullets"]]
        assert len(ids) == len(set(ids)) > 0

        # Phase 3 from the JSON files only (no PDF / PPTX), then a repeat run: identical output
        again = tmp / "p3" / "draft2.json"
        cmd = base + ["--parsed", str(tmp / "p1" / "paper_parsed.json"), "--qpp", str(tmp / "p1" / "deck_qpp.json"),
                      "--plan-json", str(plan_p), "--draft", "--draft-out", str(again)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        d2 = json.loads(again.read_text(encoding="utf-8"))
        assert [s["bullets"] for s in d2["slides"]] == [s["bullets"] for s in draft["slides"]]

        # clear errors, non-zero exit, no output file
        bad = subprocess.run(cmd[:-1] + [str(tmp / "x.json"), "--lambda-mmr", "1.5"], capture_output=True, text=True)
        assert bad.returncode == 1 and "lambda_mmr must be in [0, 1]" in bad.stderr and not (tmp / "x.json").exists()
        noplan = subprocess.run([a for a in cmd if a not in ("--plan-json", str(plan_p))], capture_output=True, text=True)
        assert noplan.returncode == 1 and "needs a Phase 2 plan" in noplan.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"ALL PHASE 3 TESTS PASSED ({len(tests)} tests)")

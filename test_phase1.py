"""
Basic end-to-end test for Phase 1.   Run:   python tests/test_phase1.py
(also collectable by pytest).  Generates synthetic data, runs main.py as a
subprocess, then checks the JSON files and the console output.
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

from make_synthetic_data import make_pdf, make_pptx   # noqa: E402
from parser import split_sentences                    # noqa: E402
from section_tagger import tag_title                  # noqa: E402

SECTIONS = ["Motivation", "Background", "Method", "Setup", "Results", "Conclusion"]


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_unit():
    assert tag_title("Experimental Results") == "Results"       # keyword collision from the proposal
    assert tag_title("Experimental Setup") == "Setup"
    assert tag_title("Totally Unmatched") is None
    s = split_sentences("We follow Knuth et al. closely, e.g. in Fig. 3. Accuracy rises from 0.76 to 0.87. Done here.")
    assert s == ["We follow Knuth et al. closely, e.g. in Fig. 3.", "Accuracy rises from 0.76 to 0.87.", "Done here."], s


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf, pptx, out = tmp / "paper.pdf", tmp / "deck.pptx", tmp / "out"
        make_pdf(pdf)
        make_pptx(pptx)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "src" / "main.py"), "--paper", str(pdf), "--deck", str(pptx), "--out", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        for needle in ("[1] PAPER PARSE", "[2] QUANTIFIED PREFERENCE PROFILE", "sum(w) = 1.000000", "pi (flow)"):
            assert needle in proc.stdout, f"console output missing {needle!r}"

        # ---------------- parsed paper JSON ---------------- #
        paper = json.loads((out / "paper_parsed.json").read_text(encoding="utf-8"))
        assert paper["title"] == "Faster Widgets via Learned Sorting"
        assert paper["n_pages"] == 2
        heads = {s["heading"]: s["label"] for s in paper["sections"]}
        assert heads["Experimental Results"] == "Results"
        assert heads["Experimental Setup"] == "Setup"
        assert heads["Related Work"] == "Background"
        assert heads["Model Architecture"] == "Method"
        assert paper["present_labels_P"] == SECTIONS
        texts = " ".join(s["text"] for s in paper["sentences"])
        assert "extractive feature encoder" in texts                # hyphenation repaired
        assert "Synthetic Test Paper" not in texts                  # running header removed
        assert "Figure 1" not in texts and "Addison-Wesley" not in texts     # caption and references removed
        assert all({"id", "page", "section", "label", "text"} <= s.keys() for s in paper["sentences"])
        assert [s["id"] for s in paper["sentences"]] == list(range(len(paper["sentences"])))
        assert paper["abstract"].startswith("We study widget sorting")

        # ---------------- QPP JSON ---------------- #
        q = json.loads((out / "deck_qpp.json").read_text(encoding="utf-8"))
        th = q["theta"]
        assert q["n_slides_total"] == 12 and q["n_content_slides"] == 9
        assert {e["reason"] for e in q["excluded_slides"]} == {"title", "outline", "closing"}
        exp_counts = [1, 1, 2, 1, 3, 1]
        for k, c in zip(SECTIONS, exp_counts):
            assert q["section_slide_counts"][k] == c, (k, q["section_slide_counts"])
            assert approx(th["w"][k], c / 9, 1e-5)
        assert approx(sum(th["w"].values()), 1.0, 1e-5)
        assert th["pi"] == SECTIONS
        phi = th["phi"]
        # beta: >= 3 body paragraphs of ANY kind (3,4,3,4,3,3 -> 6 of 9 slides); prose counts here
        assert approx(phi["beta"], 6 / 9, 1e-5)
        # rho: picture OR table only -> slides 6, 8, 9; the chart-only slide 10 must NOT count
        assert approx(phi["rho"], 3 / 9, 1e-5)
        assert approx(phi["epsilon"], 1 / 9, 1e-5)      # one equation slide
        # b_bar: REAL bullet paragraphs / content slides (2+4+3+0+4+0+0+0+3 = 16); prose is not a bullet
        assert q["total_bullet_paragraphs"] == 16
        assert approx(q["mean_bullets"], 16 / 9, 1e-5)
        assert approx(q["mean_body_paragraphs"], 21 / 9, 1e-5)   # diagnostic: differs from b_bar on purpose
        assert not approx(q["mean_bullets"], q["mean_body_paragraphs"], 1e-3)
        by_title = {s["title"]: s for s in q["slides"]}
        intro = by_title["Introduction"]                         # body placeholder with one <a:buNone/> paragraph
        assert (intro["n_body_paragraphs"], intro["n_bullet_paragraphs"]) == (3, 2)
        keynum = by_title["Key Numbers"]                         # 3 prose paragraphs in a plain text box + chart
        assert (keynum["n_body_paragraphs"], keynum["n_bullet_paragraphs"]) == (3, 0)
        assert keynum["n_charts"] == 1 and keynum["has_figure"] is False
        res = by_title["Experimental Results"]                   # table + one prose note
        assert res["n_tables"] == 1 and res["has_figure"] is True and res["n_bullet_paragraphs"] == 0
        assert by_title["Related Work"]["n_bullet_paragraphs"] == 4
        assert 0.0 < phi["eta"] < 2.0
        inherited = [s["title"] for s in q["slides"] if s["label_source"] == "inherited"]
        assert inherited == ["Block Diagram", "Key Numbers"], inherited
    print("ALL PHASE 1 TESTS PASSED")


if __name__ == "__main__":
    test_unit()
    test_end_to_end()

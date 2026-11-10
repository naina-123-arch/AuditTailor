"""
Generate a synthetic paper PDF and a synthetic reference deck PPTX for the Phase 1 test.

The PDF includes: a title, authors, abstract, numbered sections and a subsection,
"Experimental Results" (keyword-collision case), running header + page numbers,
a figure caption, a hyphenated line break, abbreviations (et al., e.g.) and a
reference list.  The deck has a title slide, an outline slide, 9 content slides
(2 without a keyword title, to exercise inheritance), a table, pictures, an
equation, a prose-only slide (3 paragraphs, 0 bullets), a no-bullet paragraph
inside a body placeholder, a chart (must not count toward rho), and a closing slide.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
class _Writer:
    W, H, LM, RM, TOP, BOT = 595, 842, 72, 72, 90, 760

    def __init__(self):
        self.doc = fitz.open()
        self.page = None
        self.y = 0
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page(width=self.W, height=self.H)
        self.y = self.TOP

    def _ensure(self, h):
        if self.y + h > self.BOT:
            self._new_page()

    def heading(self, text, size=13):
        self._ensure(40)
        self.y += 10
        self.page.insert_text((self.LM, self.y), text, fontsize=size, fontname="hebo")
        self.y += 18

    def para(self, text=None, lines=None, size=10):
        width = self.W - self.LM - self.RM
        if lines is None:
            lines, cur = [], ""
            for word in text.split():
                cand = (cur + " " + word).strip()
                if fitz.get_text_length(cand, fontname="helv", fontsize=size) <= width:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
        for ln in lines:
            self._ensure(14)
            self.page.insert_text((self.LM, self.y), ln, fontsize=size, fontname="helv")
            self.y += 12.5
        self.y += 6

    def caption(self, text):
        self.para(text, size=9)

    def finish(self, path):
        n = len(self.doc)
        for i, pg in enumerate(self.doc, 1):
            pg.insert_text((self.LM, 40), "Synthetic Test Paper - AuditTailor Phase 1", fontsize=8, fontname="helv")
            pg.insert_text((self.W / 2 - 4, 810), str(i), fontsize=9, fontname="helv")
        self.doc.save(str(path))
        self.doc.close()
        return n


def make_pdf(path: Path) -> None:
    w = _Writer()
    pg = w.page
    pg.insert_text((120, 110), "Faster Widgets via Learned Sorting", fontsize=20, fontname="hebo")
    pg.insert_text((190, 140), "A. Author and B. Writer", fontsize=11, fontname="helv")
    w.y = 175
    w.heading("Abstract", 12)
    w.para("We study widget sorting and propose a learned sorter that is faster than classical baselines. "
           "Accuracy rises from 0.76 to 0.87 on the benchmark suite.")
    w.heading("1 Introduction")
    w.para("Sorting widgets is a core step in many pipelines and it is often the bottleneck. "
           "Existing methods ignore the structure of the data, e.g. nearly sorted inputs. "
           "We therefore ask whether a learned model can exploit that structure.")
    w.para("Our contributions are threefold and are summarised in the following sections of this paper.")
    w.heading("2 Related Work")
    w.para("Classical comparison sorts were analysed by Knuth et al. in great detail. "
           "Learned index structures were introduced more recently and reduce lookup time. "
           "Hybrid approaches combine both ideas but require manual tuning of many parameters.")
    w.heading("3 Proposed Method")
    w.para(lines=["Our method first computes an embedding of each widget with an extrac-",
                  "tive feature encoder that runs in linear time over the input stream.",
                  "The embedding is then bucketed by a small neural predictor."])
    w.heading("3.1 Model Architecture", 11.5)
    w.para("The predictor has two hidden layers and a softmax output over buckets. "
           "Buckets are merged greedily until each one fits in cache. "
           "The final pass is an insertion sort over nearly ordered buckets.")
    w.caption("Figure 1: Overview of the learned sorter. The encoder feeds the bucket predictor.")
    w.heading("4 Experimental Setup")
    w.para("We evaluate on three synthetic datasets of one million widgets each. "
           "Baselines are quicksort, mergesort and a learned index. "
           "All runs use a single core and five random seeds.")
    w.heading("5 Experimental Results")
    w.para("Accuracy rises from 0.76 to 0.87 when the predictor is enabled. "
           "Runtime drops by 31 percent relative to quicksort on nearly sorted inputs. "
           "An ablation shows that removing the encoder hurts accuracy by 4 points.")
    w.heading("6 Conclusion")
    w.para("We presented a learned sorter that is both faster and more accurate than the baselines. "
           "Future work will extend it to streaming data.")
    w.heading("References", 13)
    w.para("[1] D. Knuth. The Art of Computer Programming, Volume 3. Addison-Wesley, 1998.")
    w.para("[2] T. Kraska et al. The case for learned index structures. SIGMOD 2018.")
    w.finish(path)


# --------------------------------------------------------------------------- #
# PPTX
# --------------------------------------------------------------------------- #
def _png_bytes() -> io.BytesIO:
    img = Image.new("RGB", (240, 140), (200, 220, 245))
    ImageDraw.Draw(img).rectangle((20, 20, 220, 120), outline=(30, 60, 140), width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def make_pptx(path: Path) -> None:
    prs = Presentation()                      # default 4:3 template
    L_TITLE, L_CONTENT, L_TITLE_ONLY = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[5]

    def content(title, bullets, font=None):
        s = prs.slides.add_slide(L_CONTENT)
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.text = bullets[0]
        for b in bullets[1:]:
            tf.add_paragraph().text = b
        if font:
            for p in tf.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font)
        return s

    def no_bullet(paragraph):
        """Explicitly switch the bullet off (<a:buNone/>): ordinary prose inside a body placeholder."""
        pPr = paragraph._p.get_or_add_pPr()
        etree.SubElement(pPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}buNone")

    def title_only(title, picture=False, table=False, note=None):
        s = prs.slides.add_slide(L_TITLE_ONLY)
        s.shapes.title.text = title
        if picture:
            s.shapes.add_picture(_png_bytes(), Inches(2.5), Inches(2.0), Inches(5), Inches(3))
        if table:
            t = s.shapes.add_table(3, 3, Inches(1), Inches(2), Inches(8), Inches(2)).table
            for r in range(3):
                for c in range(3):
                    t.cell(r, c).text = f"{r}{c}"
        if note:
            tb = s.shapes.add_textbox(Inches(1), Inches(5.5), Inches(8), Inches(0.8))
            tb.text_frame.word_wrap = True
            tb.text_frame.text = note
        return s

    # 1 title
    s = prs.slides.add_slide(L_TITLE)
    s.shapes.title.text = "Faster Widgets via Learned Sorting"
    s.placeholders[1].text = "A. Author"
    # 2 outline (excluded)
    content("Outline", ["Motivation", "Method", "Results"])
    # 3..11 content (9 slides)
    s3 = content("Introduction", ["Sorting is a bottleneck", "Data has structure", "Can we learn it?"], 24)      # Motivation: 3 par, 2 bullets
    no_bullet(s3.placeholders[1].text_frame.paragraphs[2])                                                     # 3rd paragraph is prose, not a bullet
    content("Related Work", ["Comparison sorts", "Learned indexes", "Hybrid approaches", "Manual tuning"], 24)   # Background, 4 par
    content("Proposed Method", ["Embed widgets", "Predict bucket"], 24).placeholders[1].text_frame.add_paragraph().text = "L = a + b * c"  # Method, 3 par, equation
    title_only("Block Diagram", picture=True)                                                                  # no keyword -> inherits Method, figure
    content("Experimental Setup", ["Three datasets", "Four baselines", "Five seeds", "Single core"], 24)       # Setup, 4 par
    title_only("Experimental Results", table=True, note="Accuracy 0.76 to 0.87")                               # Results, table
    title_only("Ablation Analysis", picture=True)                                                              # Results, figure
    s10 = title_only("Key Numbers")                                                                            # no keyword -> inherits Results
    tb = s10.shapes.add_textbox(Inches(1), Inches(1.5), Inches(8), Inches(1.6))                                # 3 PROSE paragraphs (no bullets)
    tb.text_frame.word_wrap = True
    prose = ["The learned sorter is much faster on nearly sorted inputs.",
             "It stays competitive on random inputs.",
             "Memory use is unchanged compared with the baselines."]
    tb.text_frame.text = prose[0]
    for t in prose[1:]:
        tb.text_frame.add_paragraph().text = t
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(16)
    cd = CategoryChartData()                                                                                   # a CHART: must not count toward rho
    cd.categories = ["quick", "merge", "learned"]
    cd.add_series("runtime", (1.0, 1.2, 0.69))
    s10.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(2), Inches(3.3), Inches(6), Inches(3.5), cd)
    content("Conclusion", ["Learned sorting works", "Future: streaming", "Thank the reviewers"], 24)           # Conclusion, 3 par
    # 12 closing (excluded)
    content("Thank You", ["Questions?"])
    prs.save(str(path))


if __name__ == "__main__":
    (ROOT / "data" / "papers").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "reference_decks").mkdir(parents=True, exist_ok=True)
    make_pdf(ROOT / "data" / "papers" / "synthetic_paper.pdf")
    make_pptx(ROOT / "data" / "reference_decks" / "synthetic_reference.pptx")
    print("synthetic data written")

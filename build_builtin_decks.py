"""
build_builtin_decks.py -- (re)generate the bundled reference presentations in data/reference_decks/builtin/.

    python tools/build_builtin_decks.py

These decks are NOT real conference talks and carry no scientific content.  Each is a small, generic PPTX whose
*structure* (slides per section, section order, bullets per slide, figure/table/equation usage, text density)
differs by style, so that Phase 1 (QPP extraction) can read a presentation profile from it.  The same file is also
the Phase 4/6 visual template (the project's existing behaviour: the reference deck supplies master + layouts).
The styles are simply bundled profiles; they are not scientifically validated.

Deterministic: fixed content, no randomness (only the ZIP timestamps inside the .pptx differ between runs).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reference_decks" / "builtin"

# slide titles per canonical section (each must hit the section keyword table of src/section_tagger.py)
TITLES = {
    "Motivation": ["Introduction", "Problem Statement"],
    "Background": ["Background", "Related Work", "Preliminaries"],
    "Method": ["Proposed Method", "Model Architecture", "Algorithm", "Method Details", "Method Components"],
    "Setup": ["Experimental Setup", "Datasets", "Baselines", "Implementation Details"],
    "Results": ["Main Results", "Ablation Study", "Further Results", "Qualitative Results", "Result Comparison"],
    "Conclusion": ["Conclusion", "Future Work"],
}
# generic (content-free) bullet wording: (short, normal, long)
BULLETS = {
    "Motivation": [("Why it matters", "State the problem and why it matters", "State the problem the paper addresses and explain why it matters to the field"),
                   ("Current gap", "Point out the gap in current work", "Point out the gap in current approaches that the paper sets out to close"),
                   ("Our idea", "Summarise the main idea", "Summarise the main idea of the paper and how it differs from earlier work"),
                   ("Contributions", "List the key contributions", "List the key contributions of the paper in order of importance"),
                   ("Scope", "Define the scope", "Define the scope of the study and the assumptions that are made"),
                   ("Audience", "Say who benefits", "Say which communities or applications benefit from the results")],
    "Background": [("Key terms", "Define the key terms", "Define the key terms and notation used throughout the talk"),
                   ("Prior work", "Summarise prior work", "Summarise the most relevant prior work and its main limitations"),
                   ("Baselines", "Introduce standard approaches", "Introduce the standard approaches that the new method is compared with"),
                   ("Limits", "Note their limitations", "Note the limitations of existing approaches that motivate the new method"),
                   ("Notation", "Fix the notation", "Fix the notation and the problem formulation used in later slides"),
                   ("Gap", "Link to the gap", "Link the background to the gap identified at the start of the talk")],
    "Method": [("Main idea", "Describe the main idea", "Describe the main idea of the method in a few plain sentences"),
               ("Components", "Name the main components", "Name the main components of the method and how they are connected"),
               ("Inputs", "Explain inputs and outputs", "Explain the inputs and outputs of each component of the method"),
               ("Objective", "State the objective", "State the objective that the method optimises and why it is suitable"),
               ("Steps", "Walk through the steps", "Walk through the steps of the procedure from input to final output"),
               ("Choices", "Justify design choices", "Justify the main design choices and mention alternatives that were considered")],
    "Setup": [("Data", "Describe the data", "Describe the data that is used and how it was prepared for the experiments"),
              ("Baselines", "List the baselines", "List the baselines and explain why each of them was selected"),
              ("Metrics", "Define the metrics", "Define the evaluation metrics and explain what a better value means"),
              ("Settings", "Give the settings", "Give the main settings, such as hardware, random seeds and training details"),
              ("Protocol", "Explain the protocol", "Explain the evaluation protocol including splits and repetitions"),
              ("Fairness", "Ensure a fair comparison", "Explain how a fair comparison between all methods was ensured")],
    "Results": [("Main finding", "State the main finding", "State the main finding and point to the evidence that supports it"),
                ("Comparison", "Compare with baselines", "Compare the method with the baselines using the agreed metrics"),
                ("Ablation", "Show what each part adds", "Show what each component of the method adds in the ablation study"),
                ("Analysis", "Analyse the behaviour", "Analyse when the method works well and when it does not work well"),
                ("Robustness", "Report robustness", "Report how robust the results are across datasets and random seeds"),
                ("Caveats", "Mention caveats", "Mention caveats that readers should keep in mind when using the results")],
    "Conclusion": [("Summary", "Summarise the paper", "Summarise the paper and its main results in a few sentences"),
                   ("Takeaway", "Give the key takeaway", "Give the key takeaway that the audience should remember"),
                   ("Limits", "State the limitations", "State the limitations of the work honestly and clearly"),
                   ("Future", "Outline future work", "Outline directions for future work that follow from the results"),
                   ("Thanks", "Invite questions", "Invite questions and point to the code or data if available"),
                   ("Contact", "Share contact details", "Share contact details for further discussion")],
}
EQUATION = "L = \u2212\u2211 y log p(y | x)"
SECTIONS = ["Motivation", "Background", "Method", "Setup", "Results", "Conclusion"]

# style -> (title, slides per section, bullets per content slide, wording index, body font pt,
#           {(section, k): "pic" | "table" | "eq"} for the k-th slide (0-based) of the section)
STYLES = {
    "academic_research": ("Academic Research", dict(zip(SECTIONS, (2, 2, 3, 2, 3, 2))), 4, 1, 24,
                          {("Method", 1): "pic", ("Method", 2): "eq", ("Results", 0): "table", ("Results", 1): "pic"}),
    "technical_ai": ("Technical / AI", dict(zip(SECTIONS, (1, 2, 5, 2, 4, 1))), 4, 1, 22,
                     {("Method", 0): "pic", ("Method", 1): "eq", ("Method", 2): "eq", ("Method", 3): "pic", ("Method", 4): "eq",
                      ("Results", 0): "table", ("Results", 1): "table", ("Results", 2): "pic"}),
    "conference": ("Conference", dict(zip(SECTIONS, (2, 1, 3, 1, 5, 1))), 3, 0, 26,
                   {("Method", 1): "pic", ("Method", 2): "eq", ("Results", 0): "table", ("Results", 1): "pic", ("Results", 3): "pic"}),
    "minimal_professional": ("Minimal Professional", dict(zip(SECTIONS, (1, 1, 2, 1, 2, 1))), 2, 0, 28,
                             {("Method", 1): "pic", ("Results", 0): "table"}),
    "detailed_research": ("Detailed Research", dict(zip(SECTIONS, (2, 3, 4, 3, 4, 2))), 6, 2, 18,
                          {("Method", 1): "pic", ("Method", 2): "eq", ("Results", 0): "table", ("Results", 2): "pic"}),
}


def _png() -> io.BytesIO:
    img = Image.new("RGB", (320, 200), (222, 232, 246))
    d = ImageDraw.Draw(img)
    d.rectangle((16, 16, 304, 184), outline=(40, 70, 140), width=4)
    d.line((16, 184, 304, 16), fill=(40, 70, 140), width=3)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def build(style_id: str, path: Path) -> None:
    label, counts, n_b, wording, font, extras = STYLES[style_id]
    prs = Presentation()                                   # default master: Title Slide / Title and Content / Title Only ...
    L_TITLE, L_CONTENT, L_TONLY = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[5]
    prs.core_properties.title = f"AuditTailor built-in reference: {label}"
    prs.core_properties.author = "AuditTailor"
    prs.core_properties.subject = "Generic reference presentation used only to extract presentation preferences"

    s = prs.slides.add_slide(L_TITLE)
    s.shapes.title.text = "Presentation Title"
    s.placeholders[1].text = "Author \u00b7 Affiliation"
    s = prs.slides.add_slide(L_CONTENT)                    # outline slide (excluded from the profile by the tagger)
    s.shapes.title.text = "Outline"
    tf = s.placeholders[1].text_frame
    tf.text = "Motivation"
    for sec in ("Method", "Results", "Conclusion"):
        tf.add_paragraph().text = sec

    for sec in SECTIONS:
        for k in range(counts[sec]):
            title = TITLES[sec][k % len(TITLES[sec])]
            kind = extras.get((sec, k))
            if kind == "table":
                s = prs.slides.add_slide(L_TONLY)
                s.shapes.title.text = title
                t = s.shapes.add_table(3, 3, Inches(0.8), Inches(1.8), Inches(8.4), Inches(2.4)).table
                for r in range(3):
                    for c in range(3):
                        t.cell(r, c).text = "Method" if (r, c) == (0, 0) else ("Metric" if r == 0 else "\u2014")
                tb = s.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(8.4), Inches(0.9))
                tb.text_frame.word_wrap = True
                tb.text_frame.text = BULLETS[sec][0][wording]
                for r in tb.text_frame.paragraphs[0].runs:
                    r.font.size = Pt(font)
                continue
            s = prs.slides.add_slide(L_CONTENT)
            s.shapes.title.text = title
            body = s.placeholders[1]
            lines = [BULLETS[sec][(k + i) % len(BULLETS[sec])][wording] for i in range(n_b)]
            if kind == "eq":
                lines[-1] = EQUATION
            body.text_frame.text = lines[0]
            for ln in lines[1:]:
                body.text_frame.add_paragraph().text = ln
            for p in body.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font)
            if kind == "pic":                              # text on the left, figure placeholder on the right
                body.width = Inches(5.2)
                s.shapes.add_picture(_png(), Inches(5.9), Inches(2.0), Inches(3.6), Inches(2.25))

    s = prs.slides.add_slide(L_CONTENT)                    # closing slide (excluded by the tagger)
    s.shapes.title.text = "Thank You"
    s.placeholders[1].text = "Questions?"
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(path))


def main() -> int:
    for sid in STYLES:
        build(sid, OUT / f"{sid}.pptx")
        print("wrote", (OUT / f"{sid}.pptx").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

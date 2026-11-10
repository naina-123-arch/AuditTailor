"""
qpp.py -- AuditTailor Phase 1, Stage 2 (Quantified Preference Profile)

Turns a reference .pptx deck into theta = (w, pi, phi) following Sec. 4.2:

  w    emphasis distribution over the 6 canonical sections (slide shares)
  pi   section labels in order of first appearance (the flow)
  phi  = (eta, beta, rho, epsilon), averaged over CONTENT slides
         eta     fill ratio: estimated rendered text height / text-box height,
                 averaged over text boxes
         beta    fraction of content slides with >= 3 separate body paragraphs
         rho     fraction of content slides with >= 1 picture or table
                 (strict proposal definition: charts do NOT count)
         epsilon fraction of content slides with an equation (heuristic)
  b_bar  mean number of BULLETED paragraphs per content slide
         = (bullet paragraphs over all content slides) / (number of content slides).
         A paragraph is a bullet only if PowerPoint's paragraph properties
         (resolved through the shape / layout / master inheritance chain) say so;
         plain prose paragraphs are not bullets.  It is NOT part of the style
         distance and it is separate from beta (>= 3 paragraphs of any kind).

Title, outline, reference, closing (thank-you) and section-divider slides are
excluded.  A content slide whose title matches no keyword inherits the label of
the previous content slide.

Deterministic: python-pptx + Pillow font metrics (constant-width fallback).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

try:
    from PIL import ImageFont
except ImportError:  # Pillow is optional
    ImageFont = None  # type: ignore

from section_tagger import CANONICAL_SECTIONS, excluded_kind, tag_title

# rho follows the proposal strictly: picture OR table.  Charts are only recorded
# as a separate diagnostic (SlideInfo.n_charts) and never affect rho.

BULLET_MIN_PARAGRAPHS = 3          # beta threshold from the proposal
DEFAULT_FONT_PT = 18.0
LINE_HEIGHT_FACTOR = 1.2
BODY_SPACE_BEFORE = 0.2            # PowerPoint default spcBef (20 % of font size) in body placeholders

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_NS = {"a": _A, "p": _P}

_IGNORED_PH = {
    PP_PLACEHOLDER.SLIDE_NUMBER, PP_PLACEHOLDER.FOOTER, PP_PLACEHOLDER.DATE,
    getattr(PP_PLACEHOLDER, "HEADER", PP_PLACEHOLDER.FOOTER),
}
_TITLE_PH = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE}
_BODY_PH = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT}

# --------------------------------------------------------------------------- #
# Equation heuristic ("a simple pattern for math symbols and equation objects")
# --------------------------------------------------------------------------- #
_MATH_SYMBOLS = re.compile("[\u2211\u220f\u222b\u221a\u2202\u2207\u2264\u2265\u2248\u2260\u2208\u2209\u2282\u2286\u222a\u2229\u00b1\u221e\u2200\u2203]")
_LATEX = re.compile(r"\\(?:frac|sum|int|alpha|beta|theta|lambda|mathbb|mathcal|sqrt|cdot|times|leq|geq)\b|\$[^$\n]{2,}\$")
_EQ_PATTERN = re.compile(
    r"[A-Za-z\u0370-\u03ff][\w']*(?:\([^)]{0,25}\))?\s*=\s*[^\n=]*?[\w)\]]\s*[+\-*/^]\s*[\w(\[]"
)


def text_has_equation(text: str) -> bool:
    return bool(_MATH_SYMBOLS.search(text) or _LATEX.search(text) or _EQ_PATTERN.search(text))


# --------------------------------------------------------------------------- #
# Text-fit estimator (fill ratio eta)
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 1.0, "LiberationSans"),
    ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", 1.0, "LiberationSans"),
    ("/Library/Fonts/Arial.ttf", 1.0, "Arial"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 1.0, "Arial"),
    ("C:/Windows/Fonts/arial.ttf", 1.0, "Arial"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0.90, "DejaVuSans(x0.90)"),
    ("DejaVuSans.ttf", 0.90, "DejaVuSans(x0.90)"),
    ("arial.ttf", 1.0, "Arial"),
]


class TextMeasurer:
    """Width of a string in points at a given size; Pillow metrics or a constant-width fallback."""

    def __init__(self) -> None:
        self._path: Optional[str] = None
        self._scale = 1.0
        self.name = "constant-width(0.5em)"
        self._cache: Dict[int, "ImageFont.FreeTypeFont"] = {}
        if ImageFont is not None:
            for path, scale, name in _FONT_CANDIDATES:
                try:
                    ImageFont.truetype(path, 10)
                except Exception:
                    continue
                self._path, self._scale, self.name = path, scale, f"pillow:{name}"
                break

    def width(self, text: str, size_pt: float) -> float:
        if self._path is None:
            return 0.5 * size_pt * len(text)
        key = max(1, int(round(size_pt)))
        font = self._cache.get(key)
        if font is None:
            font = ImageFont.truetype(self._path, key)
            self._cache[key] = font
        return font.getlength(text) * self._scale * (size_pt / key)

    def wrap_lines(self, text: str, size_pt: float, max_w: Optional[float]) -> int:
        """Greedy word wrap; returns the number of rendered lines (>= 1)."""
        if max_w is None:
            return 1
        max_w = max(max_w, size_pt)        # avoid pathological zero widths
        lines, cur = 1, ""
        for word in text.split():
            cand = word if not cur else cur + " " + word
            if self.width(cand, size_pt) <= max_w:
                cur = cand
                continue
            if cur:
                lines += 1
            w = self.width(word, size_pt)
            if w > max_w:                  # a single over-long word is split across lines
                extra = int(w // max_w)
                lines += extra
                cur = word[-max(1, int(len(word) * (w - extra * max_w) / w)):]
            else:
                cur = word
        return lines


# --------------------------------------------------------------------------- #
# Font-size / geometry resolution through the placeholder inheritance chain
# --------------------------------------------------------------------------- #
def _xpath_first(el, expr: str) -> Optional[str]:
    try:
        res = el.xpath(expr)
    except Exception:
        return None
    return str(res[0]) if res else None


def _lvl_sz(sp_el, level: int) -> Optional[float]:
    v = _xpath_first(sp_el, f"./p:txBody/a:lstStyle/a:lvl{level + 1}pPr/a:defRPr/@sz")
    return float(v) / 100.0 if v else None


def _master_sz(shape, level: int, ph_type) -> Optional[float]:
    style = "titleStyle" if ph_type in _TITLE_PH else ("bodyStyle" if ph_type in _BODY_PH else "otherStyle")
    try:
        master = shape.part.slide.slide_layout.slide_master
    except Exception:
        return None
    v = _xpath_first(master._element, f"./p:txStyles/p:{style}/a:lvl{level + 1}pPr/a:defRPr/@sz")
    return float(v) / 100.0 if v else None


def _placeholder_type(shape):
    try:
        return shape.placeholder_format.type if shape.is_placeholder else None
    except Exception:
        return None


def resolve_font_size(shape, paragraph, run) -> float:
    if run is not None and run.font.size:
        return run.font.size.pt
    if paragraph.font.size:
        return paragraph.font.size.pt
    level = paragraph.level or 0
    sz = _lvl_sz(shape._element, level)
    if sz:
        return sz
    ph_type = _placeholder_type(shape)
    if shape.is_placeholder:
        base = getattr(shape, "_base_placeholder", None)
        hops = 0
        while base is not None and hops < 3:
            sz = _lvl_sz(base._element, level)
            if sz:
                return sz
            base = getattr(base, "_base_placeholder", None)
            hops += 1
        sz = _master_sz(shape, level, ph_type)
        if sz:
            return sz
    return DEFAULT_FONT_PT


def _autofit_scale(shape) -> float:
    v = _xpath_first(shape._element, "./p:txBody/a:bodyPr/a:normAutofit/@fontScale")
    return float(v) / 100000.0 if v else 1.0


_BULLET_TAGS = {"buNone": False, "buChar": True, "buAutoNum": True, "buBlip": True}


def _bullet_in_pPr(pPr) -> Optional[bool]:
    """True/False if this paragraph-properties element states a bullet setting, else None."""
    if pPr is None:
        return None
    for child in pPr:
        if isinstance(child.tag, str):
            name = child.tag.rsplit("}", 1)[-1]
            if name in _BULLET_TAGS:
                return _BULLET_TAGS[name]
    return None


def _bullet_at(el, expr: str) -> Optional[bool]:
    try:
        nodes = el.xpath(expr)
    except Exception:
        return None
    return _bullet_in_pPr(nodes[0]) if nodes else None


def paragraph_has_bullet(shape, paragraph) -> bool:
    """
    Is this paragraph a real bullet?  Resolution order (first explicit setting wins):
      1. the paragraph's own <a:pPr>   (buChar / buAutoNum / buBlip = bullet, buNone = no bullet)
      2. the shape's own list style    (<a:lstStyle><a:lvlNpPr>)
      3. the layout placeholder, then the master placeholder
      4. the master text styles (titleStyle / bodyStyle / otherStyle)
    A paragraph with no explicit setting anywhere (e.g. a plain text box) is NOT a bullet.
    """
    r = _bullet_in_pPr(paragraph._p.pPr)
    if r is not None:
        return r
    level = (paragraph.level or 0) + 1
    r = _bullet_at(shape._element, f"./p:txBody/a:lstStyle/a:lvl{level}pPr")
    if r is not None:
        return r
    if not shape.is_placeholder:
        return False
    base, hops = getattr(shape, "_base_placeholder", None), 0
    while base is not None and hops < 3:
        r = _bullet_at(base._element, f"./p:txBody/a:lstStyle/a:lvl{level}pPr")
        if r is not None:
            return r
        base, hops = getattr(base, "_base_placeholder", None), hops + 1
    ph_type = _placeholder_type(shape)
    style = "titleStyle" if ph_type in _TITLE_PH else ("bodyStyle" if ph_type in _BODY_PH else "otherStyle")
    try:
        master = shape.part.slide.slide_layout.slide_master
    except Exception:
        return False
    r = _bullet_at(master._element, f"./p:txStyles/p:{style}/a:lvl{level}pPr")
    return bool(r)


def estimate_fill(shape, measurer: TextMeasurer) -> Optional[float]:
    """eta_b = estimated rendered text height / inner text-box height (None if not computable)."""
    tf = shape.text_frame
    try:
        box_w, box_h = int(shape.width), int(shape.height)
    except Exception:
        return None
    if box_w <= 0 or box_h <= 0:
        return None
    in_w = (box_w - int(tf.margin_left) - int(tf.margin_right)) / 12700.0
    in_h = (box_h - int(tf.margin_top) - int(tf.margin_bottom)) / 12700.0
    if in_h <= 0 or in_w <= 0:
        return None
    wrap = tf.word_wrap is not False
    scale = _autofit_scale(shape)
    is_body = _placeholder_type(shape) in _BODY_PH

    total, first = 0.0, True
    for p in tf.paragraphs:
        text = p.text
        if not text.strip():
            continue
        runs = [r for r in p.runs if r.text.strip()]
        size = max((resolve_font_size(shape, p, r) for r in runs), default=resolve_font_size(shape, p, None)) * scale
        indent = (27.0 + 31.5 * (p.level or 0)) if paragraph_has_bullet(shape, p) else 0.0
        avail = (in_w - indent) if wrap else None
        n_lines = sum(measurer.wrap_lines(seg, size, avail) for seg in text.replace("\x0b", "\n").split("\n"))
        ls = p.line_spacing
        if ls is None:
            line_h = size * LINE_HEIGHT_FACTOR
        elif isinstance(ls, float):
            line_h = size * LINE_HEIGHT_FACTOR * ls
        else:
            line_h = ls.pt
        before = p.space_before.pt if p.space_before is not None else (size * BODY_SPACE_BEFORE if is_body and not first else 0.0)
        after = p.space_after.pt if p.space_after is not None else 0.0
        total += n_lines * line_h + before + after
        first = False
    return total / in_h if total > 0 else None


# --------------------------------------------------------------------------- #
# Slide inspection
# --------------------------------------------------------------------------- #
def _iter_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(sh.shapes)
        else:
            yield sh


def _is_picture(sh) -> bool:
    return sh._element.tag.endswith("}pic")


def _has_math_object(sh) -> bool:
    try:
        from lxml import etree
        xml = etree.tostring(sh._element).decode("utf-8", "ignore")
    except Exception:
        return False
    return "oMath" in xml


@dataclass
class SlideInfo:
    index: int
    title: str
    layout: str
    kind: str                                  # 'content' or exclusion reason
    label: Optional[str] = None
    label_source: Optional[str] = None
    n_body_paragraphs: int = 0                 # non-empty body paragraphs (prose or bullet) -> beta
    n_bullet_paragraphs: int = 0               # body paragraphs that are real bullets      -> b_bar
    n_pictures: int = 0
    n_tables: int = 0
    n_charts: int = 0
    has_figure: bool = False
    has_equation: bool = False
    box_fills: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        fill = float(np.mean(self.box_fills)) if self.box_fills else None
        return {
            "index": self.index,
            "title": self.title,
            "layout": self.layout,
            "kind": self.kind,
            "label": self.label,
            "label_source": self.label_source,
            "n_body_paragraphs": self.n_body_paragraphs,
            "n_bullet_paragraphs": self.n_bullet_paragraphs,
            "n_pictures": self.n_pictures,
            "n_tables": self.n_tables,
            "n_charts": self.n_charts,
            "has_figure": self.has_figure,
            "has_equation": self.has_equation,
            "n_text_boxes": len(self.box_fills),
            "mean_fill": None if fill is None else round(fill, 4),
        }


def _inspect_slide(idx: int, slide, measurer: TextMeasurer, slide_h: int) -> SlideInfo:
    layout_name = slide.slide_layout.name or ""
    title_shape = slide.shapes.title
    leaves = list(_iter_shapes(slide.shapes))

    body_paras = 0
    bullet_paras = 0
    n_pic = n_tab = n_chart = 0
    has_eq = False
    fills: List[float] = []
    n_body_shapes = 0
    text_shapes = []

    for sh in leaves:
        ph = _placeholder_type(sh)
        if ph in _IGNORED_PH:
            continue
        if _is_picture(sh):
            n_pic += 1
        if getattr(sh, "has_table", False) and sh.has_table:
            n_tab += 1
        if getattr(sh, "has_chart", False) and sh.has_chart:
            n_chart += 1
        if _has_math_object(sh):
            has_eq = True
        if sh.has_text_frame and sh.text_frame.text.strip():
            text_shapes.append(sh)

    # title: placeholder title, else the largest-font text shape in the top 30 % of the slide
    if title_shape is not None and title_shape.has_text_frame:
        title_text = title_shape.text_frame.text.strip()
    else:
        title_text, best = "", -1.0
        for sh in text_shapes:
            if _placeholder_type(sh) is None and sh.top is not None and sh.top < 0.3 * slide_h:
                p = sh.text_frame.paragraphs[0]
                sz = resolve_font_size(sh, p, p.runs[0] if p.runs else None)
                if sz > best:
                    best, title_text, title_shape = sz, sh.text_frame.text.strip(), sh

    for sh in text_shapes:
        is_title = title_shape is not None and sh.shape_id == title_shape.shape_id
        ph = _placeholder_type(sh)
        text = sh.text_frame.text
        if text_has_equation(text):
            has_eq = True
        eta = estimate_fill(sh, measurer)
        if eta is not None:
            fills.append(eta)
        if is_title or ph == PP_PLACEHOLDER.SUBTITLE:
            continue
        n_body_shapes += 1
        for p in sh.text_frame.paragraphs:
            if p.text.strip():
                body_paras += 1
                if paragraph_has_bullet(sh, p):
                    bullet_paras += 1

    info = SlideInfo(
        index=idx, title=title_text, layout=layout_name, kind="content",
        n_body_paragraphs=body_paras, n_bullet_paragraphs=bullet_paras, n_pictures=n_pic, n_tables=n_tab, n_charts=n_chart,
        has_equation=has_eq, box_fills=fills,
    )
    info.has_figure = bool(n_pic or n_tab)      # strict: picture OR table; charts are diagnostic only

    # ---- exclusion rules: title / outline / references / closing / divider --- #
    ex = excluded_kind(title_text)
    lname = layout_name.lower()
    if ex:
        info.kind = ex
    elif idx == 1 and ("title slide" in lname or (n_body_shapes == 0 or body_paras <= 3) and not info.has_figure and "content" not in lname):
        info.kind = "title"
    elif "section header" in lname or ("title slide" in lname):
        info.kind = "section_divider"
    elif not title_text and body_paras == 0 and not info.has_figure:
        info.kind = "empty"
    return info


# --------------------------------------------------------------------------- #
# QPP
# --------------------------------------------------------------------------- #
@dataclass
class QPP:
    source: str
    n_slides: int
    slides: List[SlideInfo]
    counts: Dict[str, int]
    w: Dict[str, float]
    pi: List[str]
    eta: float
    beta: float
    rho: float
    epsilon: float
    mean_bullets: float                        # bullet paragraphs / content slides
    total_bullets: int
    mean_paragraphs: float                     # diagnostic only (prose + bullets)
    fit_estimator: str
    warnings: List[str]

    @property
    def n_content(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> Dict:
        r = lambda x: round(float(x), 6)
        return {
            "schema": "audittailor.qpp.v1",
            "source": self.source,
            "n_slides_total": self.n_slides,
            "n_content_slides": self.n_content,
            "theta": {
                "w": {k: r(v) for k, v in self.w.items()},
                "pi": list(self.pi),
                "phi": {
                    "eta": r(self.eta),
                    "beta": r(self.beta),
                    "rho": r(self.rho),
                    "epsilon": r(self.epsilon),
                },
            },
            "mean_bullets": r(self.mean_bullets),
            "total_bullet_paragraphs": int(self.total_bullets),
            "mean_body_paragraphs": r(self.mean_paragraphs),
            "section_slide_counts": dict(self.counts),
            "excluded_slides": [
                {"index": s.index, "title": s.title, "reason": s.kind}
                for s in self.slides if s.kind != "content"
            ],
            "slides": [s.to_dict() for s in self.slides],
            "fit_estimator": self.fit_estimator,
            "warnings": list(self.warnings),
        }


def extract_qpp(pptx_path: str | Path) -> QPP:
    pptx_path = Path(pptx_path)
    if not pptx_path.is_file():
        raise FileNotFoundError(f"PPTX not found: {pptx_path}")
    prs = Presentation(str(pptx_path))
    measurer = TextMeasurer()
    slide_h = int(prs.slide_height)

    slides = [_inspect_slide(i, s, measurer, slide_h) for i, s in enumerate(prs.slides, start=1)]
    content = [s for s in slides if s.kind == "content"]
    warnings: List[str] = []

    # ---- section labels with the inheritance rule ---------------------------- #
    prev: Optional[str] = None
    for s in content:
        lab = tag_title(s.title)
        if lab:
            s.label, s.label_source = lab, "title"
        elif prev:
            s.label, s.label_source = prev, "inherited"
        else:
            s.label, s.label_source = "Motivation", "default"
            warnings.append(f"slide {s.index}: no keyword match and no previous slide; defaulted to Motivation")
        prev = s.label
    n_def = sum(1 for s in content if s.label_source != "title")
    if content and n_def:
        warnings.append(f"{n_def}/{len(content)} content slides were labelled by inheritance/default, not by title keyword")

    counts = {k: 0 for k in CANONICAL_SECTIONS}
    for s in content:
        counts[s.label] += 1
    total = sum(counts.values())
    if total == 0:
        warnings.append("no content slides found; w is all zeros and phi is zero")
        w = {k: 0.0 for k in CANONICAL_SECTIONS}
        return QPP(str(pptx_path), len(slides), slides, counts, w, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, measurer.name, warnings)

    w_vec = np.array([counts[k] for k in CANONICAL_SECTIONS], dtype=float) / total
    w = {k: float(v) for k, v in zip(CANONICAL_SECTIONS, w_vec)}

    pi: List[str] = []
    for s in content:
        if s.label not in pi:
            pi.append(s.label)

    all_fills = [f for s in content for f in s.box_fills]
    eta = float(np.mean(all_fills)) if all_fills else 0.0
    beta = float(np.mean([s.n_body_paragraphs >= BULLET_MIN_PARAGRAPHS for s in content]))
    rho = float(np.mean([s.has_figure for s in content]))
    epsilon = float(np.mean([s.has_equation for s in content]))
    total_bullets = int(sum(s.n_bullet_paragraphs for s in content))
    b_bar = total_bullets / len(content)
    mean_par = float(np.mean([s.n_body_paragraphs for s in content]))
    if not all_fills:
        warnings.append("no text boxes measured; eta = 0")
    if measurer.name.startswith("constant"):
        warnings.append("Pillow/fonts unavailable: eta uses a constant 0.5 em character width")

    return QPP(str(pptx_path), len(slides), slides, counts, w, pi, eta, beta, rho, epsilon, b_bar, total_bullets, mean_par, measurer.name, warnings)

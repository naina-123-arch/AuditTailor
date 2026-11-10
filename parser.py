"""
parser.py -- AuditTailor Phase 1, Stage 1 (Parse & tag)

Reads a scientific-paper PDF with PyMuPDF and produces, following Sec. 4.2 of
the proposal:

  * section headings, found from lines set in bold / larger type and from
    numbered-heading patterns (with a sequence check to reject figure text);
  * running headers, footers, page numbers, captions and the reference list
    removed;
  * text joined across hyphenation and split into sentences by a rule-based
    splitter;
  * every sentence stored with its page number and section, so that it can be
    cited as a pointer (page, section, sentence id);
  * every section tagged with one of the K = 6 canonical labels.

Everything here is deterministic.  No network, no LLM, no API key.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; older versions only `fitz`
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from section_tagger import CANONICAL_SECTIONS, excluded_kind, normalise_title, tag_title

# --------------------------------------------------------------------------- #
# Tunables (all deterministic constants)
# --------------------------------------------------------------------------- #
MARGIN_FRAC = 0.075            # top/bottom band searched for headers/footers
HEADING_MAX_CHARS = 90
LARGE_FONT_RATIO = 1.12        # "larger type" relative to body size
TITLE_FONT_RATIO = 1.20
MIN_SENTENCE_WORDS = 5
MIN_ALPHA_RATIO = 0.60

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00a0": " ", "\u00ad": "", "\u2009": " ", "\u202f": " ",
}

_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.|table|algorithm)\s+\d+\s*[.:\u2013-]", re.I)
_PAGE_NO_RE = re.compile(r"^\W*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?\W*$", re.I)
_NUM_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Za-z].*)$")
_ROMAN_HEADING_RE = re.compile(r"^([IVX]{1,4})\.\s+([A-Z][A-Z0-9 \-&,:]+)$")
_KNOWN_NAME_RE = re.compile(
    r"^(abstract|introduction|related works?|background|preliminar(?:y|ies)|methods?|methodology|"
    r"approach|experiments?|results?|discussion|conclusions?|references|bibliography|"
    r"acknowledg(?:e)?ments?|appendix.*|limitations|future work)$"
)
_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|et al|Figs?|Eqs?|Secs?|Tab|vs|cf|approx|resp|Dr|Mr|Ms|Prof|No|Vol|pp)\."
)
_INITIAL_RE = re.compile(r"\b[A-Z]\.(?=\s+[A-Z])")
_SENT_BOUNDARY_RE = re.compile(r"([.!?][\"')\]]*)\s+(?=[\"'(\[]*[A-Z0-9])")
_PLACEHOLDER = "\x00"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Line:
    page: int
    text: str
    size: float
    bold_frac: float
    x0: float
    y0: float
    x1: float
    y1: float
    page_h: float
    block: int


@dataclass
class Section:
    index: int
    number: Optional[str]
    heading: str
    level: int
    page: int
    label: Optional[str] = None
    label_source: Optional[str] = None
    excluded: Optional[str] = None            # 'abstract', 'references', ... or None
    chunks: List[Tuple[int, str]] = field(default_factory=list)  # (page, paragraph text)


@dataclass
class Sentence:
    id: int
    page: int
    section_index: int
    section: str
    label: str
    text: str


@dataclass
class ParsedPaper:
    path: str
    title: str
    abstract: str
    n_pages: int
    body_font_size: float
    sections: List[Section]
    sentences: List[Sentence]
    captions: List[Dict]
    removed_margin_lines: int
    dropped_fragments: int
    dropped_examples: List[str]

    # ---- derived views ---------------------------------------------------- #
    def label_word_counts(self) -> Dict[str, int]:
        """l_k: words per canonical label over body sentences (0 if absent)."""
        counts = {k: 0 for k in CANONICAL_SECTIONS}
        for s in self.sentences:
            counts[s.label] += len(s.text.split())
        return counts

    def present_labels(self) -> List[str]:
        """P = {k : l_k > 0}, in canonical order."""
        wc = self.label_word_counts()
        return [k for k in CANONICAL_SECTIONS if wc[k] > 0]

    def pointer(self, sentence_id: int) -> Tuple[int, str, int]:
        """(page, section, sentence id) -- the pointer format of Sec. 4.4."""
        s = self.sentences[sentence_id]
        return (s.page, s.section, s.id)

    def to_dict(self) -> Dict:
        wc = self.label_word_counts()
        sent_per_section: Dict[int, int] = defaultdict(int)
        words_per_section: Dict[int, int] = defaultdict(int)
        for s in self.sentences:
            sent_per_section[s.section_index] += 1
            words_per_section[s.section_index] += len(s.text.split())
        return {
            "schema": "audittailor.parsed_paper.v1",
            "source": self.path,
            "title": self.title,
            "abstract": self.abstract,
            "n_pages": self.n_pages,
            "body_font_size_pt": self.body_font_size,
            "n_sections": len([s for s in self.sections if not s.excluded]),
            "n_sentences": len(self.sentences),
            "present_labels_P": self.present_labels(),
            "label_word_counts_l": wc,
            "sections": [
                {
                    "index": s.index,
                    "number": s.number,
                    "heading": s.heading,
                    "level": s.level,
                    "page": s.page,
                    "label": s.label,
                    "label_source": s.label_source,
                    "excluded": s.excluded,
                    "n_sentences": sent_per_section.get(s.index, 0),
                    "n_words": words_per_section.get(s.index, 0),
                }
                for s in self.sections
            ],
            "sentences": [
                {
                    "id": s.id,
                    "page": s.page,
                    "section_index": s.section_index,
                    "section": s.section,
                    "label": s.label,
                    "text": s.text,
                }
                for s in self.sentences
            ],
            "captions": self.captions,
            "diagnostics": {
                "removed_margin_lines": self.removed_margin_lines,
                "dropped_fragments": self.dropped_fragments,
                "dropped_examples": self.dropped_examples,
            },
        }


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    for a, b in _LIGATURES.items():
        text = text.replace(a, b)
    return re.sub(r"[ \t]+", " ", text).strip()


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10}
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
        if i + 1 < len(s) and vals[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total


def _extract_blocks(doc) -> List[List[Line]]:
    """Return text blocks (each a list of Lines) in reading order, page by page."""
    blocks: List[List[Line]] = []
    for pno in range(len(doc)):
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        raw = page.get_text("dict")["blocks"]
        page_blocks = []
        for bi, b in enumerate(raw):
            if b.get("type", 1) != 0:
                continue
            lines: List[Line] = []
            for l in b["lines"]:
                spans = [s for s in l["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in l["spans"]))
                if not text:
                    continue
                n_chars = sum(len(s["text"].strip()) for s in spans) or 1
                size = sum(s["size"] * len(s["text"].strip()) for s in spans) / n_chars
                bold_chars = sum(
                    len(s["text"].strip())
                    for s in spans
                    if (s["flags"] & 16) or "bold" in s["font"].lower() or s["font"].lower().endswith(("-bd", ",bold"))
                )
                x0, y0, x1, y1 = l["bbox"]
                lines.append(Line(pno + 1, text, size, bold_chars / n_chars, x0, y0, x1, y1, H, bi))
            if lines:
                bx0 = min(l.x0 for l in lines)
                bx1 = max(l.x1 for l in lines)
                by0 = min(l.y0 for l in lines)
                page_blocks.append((bx0, bx1, by0, lines))

        # Simple two-column handling: if there are several narrow blocks in both
        # halves of the page, read the left column first, then the right.
        left = [b for b in page_blocks if b[1] <= 0.56 * W]
        right = [b for b in page_blocks if b[0] >= 0.44 * W and (b[1] - b[0]) < 0.6 * W]
        if len(left) >= 3 and len(right) >= 3:
            page_blocks.sort(key=lambda b: (0 if (b[0] + b[1]) / 2 < 0.5 * W else 1, b[2]))
        for *_, lines in page_blocks:
            blocks.append(lines)
    return blocks


def _is_margin(line: Line) -> bool:
    return line.y1 <= MARGIN_FRAC * line.page_h or line.y0 >= (1 - MARGIN_FRAC) * line.page_h


def _norm_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower().strip())


def _remove_margin_lines(blocks: List[List[Line]], n_pages: int) -> Tuple[List[List[Line]], int]:
    """Drop running headers/footers (repeated margin text) and page numbers."""
    pages_seen: Dict[str, set] = defaultdict(set)
    for blk in blocks:
        for l in blk:
            if _is_margin(l):
                pages_seen[_norm_repeat(l.text)].add(l.page)
    threshold = max(2, math.ceil(0.5 * n_pages))
    repeated = {k for k, v in pages_seen.items() if len(v) >= threshold}

    removed = 0
    out: List[List[Line]] = []
    for blk in blocks:
        kept = []
        for l in blk:
            if _is_margin(l) and (_norm_repeat(l.text) in repeated or _PAGE_NO_RE.match(l.text)):
                removed += 1
                continue
            kept.append(l)
        if kept:
            out.append(kept)
    return out, removed


def _body_font_size(blocks: List[List[Line]]) -> float:
    c: Counter = Counter()
    for blk in blocks:
        for l in blk:
            c[round(l.size * 2) / 2] += len(l.text)
    return c.most_common(1)[0][0] if c else 10.0


def _join_lines(lines: List[str]) -> str:
    """Join physical lines of one paragraph, repairing end-of-line hyphenation."""
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = out + " " + ln
    return re.sub(r"\s+", " ", out).strip()


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence splitter (abbreviations and initials protected)."""
    t = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    t = _INITIAL_RE.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), t)
    out, start = [], 0
    for m in _SENT_BOUNDARY_RE.finditer(t):
        out.append(t[start:m.end(1)])
        start = m.end()
    out.append(t[start:])
    return [s.replace(_PLACEHOLDER, ".").strip() for s in out if s.strip()]


def _is_body_sentence(s: str) -> bool:
    """Keep real prose sentences; drop figure text, equations, table cells, list fragments."""
    words = s.split()
    if len(words) < MIN_SENTENCE_WORDS:
        return False
    non_space = re.sub(r"\s+", "", s)
    if not non_space:
        return False
    alpha = sum(ch.isalpha() for ch in non_space) / len(non_space)
    if alpha < MIN_ALPHA_RATIO:
        return False
    if not re.search(r"[.!?][\"')\]]*$", s):
        return False
    return True


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #
class _HeadingDetector:
    """Stateful detector: remembers the last accepted numbering to reject figure text."""

    def __init__(self, body_size: float):
        self.body = body_size
        self.last_top = 0
        self.ref_size: Dict[int, float] = {}     # heading font size per level (first accepted heading)

    def detect(self, line: Line) -> Optional[Tuple[Optional[str], str, int]]:
        t = line.text.strip()
        if not (3 <= len(t) <= HEADING_MAX_CHARS):
            return None
        bold = line.bold_frac >= 0.9
        large = line.size >= self.body * LARGE_FONT_RATIO
        emphasised = bold or large

        m = _NUM_HEADING_RE.match(t)
        if m and emphasised:
            nums = tuple(int(x) for x in m.group(1).split("."))
            lvl = len(nums)
            size_ok = lvl not in self.ref_size or abs(line.size - self.ref_size[lvl]) <= 0.75
            if size_ok and self._numbering_ok(nums) and not t.endswith((".", ",", ";")):
                self.ref_size.setdefault(lvl, line.size)
                self._accept(nums)
                return (m.group(1), m.group(2).strip(), len(nums))

        m = _ROMAN_HEADING_RE.match(t)
        if m:
            n = _roman_to_int(m.group(1))
            if self._numbering_ok((n,)):
                self._accept((n,))
                return (str(n), m.group(2).strip().title(), 1)

        if emphasised and _KNOWN_NAME_RE.match(normalise_title(t)):
            return (None, t.rstrip(":"), 1)

        if line.size >= self.body * 1.25 and not t.endswith((".", ",", ";")) and len(t) <= 70:
            return (None, t, 1)
        return None

    def _numbering_ok(self, nums: Tuple[int, ...]) -> bool:
        if len(nums) == 1:
            if self.last_top == 0:
                return nums[0] in (1, 2)
            return nums[0] == self.last_top + 1   # strict: rejects numbered text inside figures
        return nums[0] == self.last_top

    def _accept(self, nums: Tuple[int, ...]) -> None:
        self.last_top = nums[0]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def parse_pdf(path: str | Path) -> ParsedPaper:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(str(path))
    try:
        n_pages = len(doc)
        meta_title = (doc.metadata or {}).get("title") or ""
        blocks = _extract_blocks(doc)
    finally:
        doc.close()
    if not blocks:
        raise ValueError(f"No extractable text in {path} (scanned PDF? OCR is out of scope for Phase 1)")

    blocks, removed_margin = _remove_margin_lines(blocks, n_pages)
    body_size = _body_font_size(blocks)

    # ---- title: leading large lines of page 1 ------------------------------ #
    title_ids = set()
    title_parts: List[str] = []
    p1 = [l for blk in blocks for l in blk if l.page == 1]
    # title (and subtitle) = the contiguous run of large lines at the top of page 1
    for l in p1:
        if l.size >= body_size * TITLE_FONT_RATIO:
            title_parts.append(l.text)
            title_ids.add(id(l))
        else:
            break
    title = " ".join(title_parts).strip() or meta_title.strip() or path.stem

    # ---- walk lines, build sections --------------------------------------- #
    detector = _HeadingDetector(body_size)
    sections: List[Section] = []
    captions: List[Dict] = []
    cur: Optional[Section] = None
    refs_reached = False

    def flush(buf: List[str], page: Optional[int]):
        if not buf or cur is None or page is None:
            return
        text = _join_lines(buf)
        if not text:
            return
        # merge with previous chunk if a paragraph continues across a block/page break
        if cur.chunks:
            prev_page, prev = cur.chunks[-1]
            if (not re.search(r"[.!?:][\"')\]]*$", prev) and text[:1].islower()) or prev.endswith("-"):
                joined = prev[:-1] + text if prev.endswith("-") and text[:1].islower() else prev + " " + text
                cur.chunks[-1] = (prev_page, joined)
                return
        cur.chunks.append((page, text))

    for blk in blocks:
        buf: List[str] = []
        buf_page: Optional[int] = None
        in_caption = False
        for line in blk:
            if refs_reached:
                break
            if id(line) in title_ids:
                continue
            hd = detector.detect(line)
            if hd is not None:
                flush(buf, buf_page)
                buf, buf_page, in_caption = [], None, False
                number, heading, level = hd
                sec = Section(index=len(sections), number=number, heading=heading, level=level, page=line.page)
                kind = excluded_kind(heading)
                if kind == "references":
                    sec.excluded = "references"
                    sections.append(sec)
                    cur = sec
                    refs_reached = True       # everything after (incl. appendices) is excluded
                    break
                if kind in ("abstract", "closing", "appendix"):
                    sec.excluded = kind
                sections.append(sec)
                cur = sec
                continue
            if _CAPTION_RE.match(line.text):
                flush(buf, buf_page)
                buf, buf_page = [], None
                in_caption = True
                captions.append({"page": line.page, "text": line.text})
                continue
            if in_caption:
                captions[-1]["text"] += " " + line.text
                continue
            if cur is None:
                continue                      # front matter before the first heading
            if buf_page is None:
                buf_page = line.page
            buf.append(line.text)
        else:
            flush(buf, buf_page)
            continue
        flush(buf, buf_page)                  # reached via break
        if refs_reached:
            break

    # ---- label sections (inheritance for headings without a keyword match) -- #
    last_label: Optional[str] = None
    label_by_level: Dict[int, str] = {}
    for s in sections:
        if s.excluded:
            continue
        lab = tag_title(s.heading)
        if lab:
            src = "heading"
        else:
            parent = label_by_level.get(s.level - 1) if s.level > 1 else None
            if parent:
                lab, src = parent, "inherited_parent"
            elif last_label:
                lab, src = last_label, "inherited_previous"
            else:
                lab, src = "Motivation", "default"
        s.label, s.label_source = lab, src
        label_by_level[s.level] = lab
        for lvl in [k for k in label_by_level if k > s.level]:
            del label_by_level[lvl]
        last_label = lab

    # ---- sentences ----------------------------------------------------------- #
    sentences: List[Sentence] = []
    abstract_parts: List[str] = []
    dropped, dropped_examples = 0, []
    for s in sections:
        if s.excluded == "abstract":
            abstract_parts.extend(t for _, t in s.chunks)
            continue
        if s.excluded:
            continue
        for page, text in s.chunks:
            for sent in split_sentences(text):
                if _is_body_sentence(sent):
                    sentences.append(Sentence(len(sentences), page, s.index, s.heading, s.label, sent))
                else:
                    dropped += 1
                    if len(dropped_examples) < 8:
                        dropped_examples.append(sent[:80])

    return ParsedPaper(
        path=str(path),
        title=title,
        abstract=" ".join(abstract_parts).strip(),
        n_pages=n_pages,
        body_font_size=body_size,
        sections=sections,
        sentences=sentences,
        captions=captions,
        removed_margin_lines=removed_margin,
        dropped_fragments=dropped,
        dropped_examples=dropped_examples,
    )

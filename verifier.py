"""
verifier.py -- AuditTailor Phase 5: deterministic verification and evaluation of a generated deck.

Inputs : parsed target paper (Phase 1 JSON), reference QPP (Phase 1 JSON), Phase 2 plan, Phase 3 draft,
         generated Phase 4 PPTX.  The PPTX is inspected directly; Phase 4's metadata is NOT trusted or read.
Output : one JSON-serialisable dict (see ``verify``).  No LLM, no API, no network, no randomness.
         This module only MEASURES.  It never edits the deck (repair is Phase 6).

Metrics (all in [0, 1]; lower is better unless noted)
-----------------------------------------------------
Dalloc   planned vs actual content-slide counts per canonical section.
             K = sections with planned_k > 0 or actual_k > 0   (sections absent from both are not "relevant")
             Dalloc = (1/K) * SUM_k  min(1, |planned_k - actual_k| / max(1, planned_k))
         Each term is capped at 1 so that Dalloc stays in [0, 1] even when a section planned with 0 slides
         receives several.  K = 0 -> Dalloc = 0.   planned = plan["slide_allocation"];
         actual = per-label content-slide counts of the generated deck's QPP (Phase 1 extractor).

Dorder   normalised Kendall-tau (inversion) distance between the plan's section order and the generated
         deck's section order pi.  U = union of both orders; over all C(|U|,2) section pairs a pair counts as
         a disagreement if (i) both sections occur in both orders and are ranked oppositely, or (ii) a section
         of the pair is missing from one of the orders (the pair cannot be ranked there).
             Dorder = disagreements / C(|U|,2)       (|U| < 2 -> 0).     0 = identical, 1 = fully reversed/disjoint.

Dstyle   mean absolute difference of the four style features phi = (eta, beta, rho, epsilon) between the
         reference QPP and the generated deck's QPP, each |difference| clamped to [0, 1]:
             Dstyle = (|eta-eta'| + |beta-beta'| + |rho-rho'| + |eps-eps'|) / 4          (weights configurable)
         ONLY the QPP style features are used: no colour, font family, animation, etc. (future work).
         w and pi are NOT part of Dstyle (they are covered by Dalloc / Dorder, so they would be counted twice);
         their distances to the reference are reported as diagnostics only (details.dstyle.diagnostic_*).

Dpref    = l1*Dalloc + l2*Dorder + l3*Dstyle, with l1+l2+l3 = 1 (configurable; defaults are implementation
         choices, not learned).   PF = 1 - Dpref.   Both clamped to [0, 1].

SR       source-support ratio = supported / total bullets of the generated PPTX.  A bullet is supported when its
         source pointer (the "[Sxxx, p.N]" marker in the slide text) resolves to a sentence of the parsed paper
         (page agrees) AND cosine(TF-IDF(bullet), TF-IDF(source sentence)) >= threshold.
         HEURISTIC LEXICAL OVERLAP, NOT semantic entailment: it cannot tell whether a re-worded or negated
         bullet is still true, only that it shares (weighted) vocabulary with its source sentence.  TF-IDF is fitted
         on all paper sentences plus all bullets (so every bullet token has a vocabulary slot and invented words
         lower the score).  No bullets -> SR = 1 (vacuous, flagged).

NF       numeric fidelity = correct / checked numeric claims.  A numeric claim is a number token in a bullet; it
         is correct when the same value occurs in the bullet's source sentence (each source occurrence can back
         one claim).  Values are normalised ("1,000" == "1000", "0.80" == "0.8").  Omitting a source number is
         not an error (clause trimming is allowed); adding or changing one is.
         HEURISTIC LIMITS: number words ("five", "one million") are not parsed; units/percent signs are ignored
         ("31 %" and "31 seconds" both read as 31); numbers glued to letters ("L2", "S000") are not claims;
         whether the number is attached to the right quantity is not checked.  No numeric bullets -> NF = 1 (vacuous).

O        overflow rate = overflowed text boxes / checked text boxes, measured on the PPTX itself.  A box is
         counted when (a) the text content, measured from the shape's real geometry / fonts / spacing in the
         generated PPTX, does not fit inside the text-frame's inner height (primary detector), or (b) any text
         run (source markers excepted) is below min_font_size.  qpp.estimate_fill is only a supplementary
         signal and never clears a positive actual-overflow finding.

J(S)     = Dpref + mu*O + nu*(1 - NF) + xi*(1 - SR).      Lower is better.   mu, nu, xi are configurable.

Non-content slides: the generated deck's QPP is computed on a copy without the slides whose titles are listed
in ``NON_CONTENT_TITLES`` (Phase 4's reference-only "Source Index" slide; Phase 1's extractor would otherwise count
it as a content slide labelled by inheritance).  Phase 1 and Phase 4 code is unchanged.
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER
from sklearn.feature_extraction.text import TfidfVectorizer

from qpp import TextMeasurer, estimate_fill, extract_qpp, paragraph_has_bullet, resolve_font_size
from section_tagger import CANONICAL_SECTIONS

SCHEMA = "audittailor.verification.v1"

# ------------------------------------------------------------------ configuration (all overridable) #
DEFAULT_PREF_WEIGHTS = {"alloc": 0.4, "order": 0.2, "style": 0.4}          # lambda_1..3, sum to 1
DEFAULT_STYLE_WEIGHTS = {"eta": 0.25, "beta": 0.25, "rho": 0.25, "epsilon": 0.25}
DEFAULT_OBJECTIVE_WEIGHTS = {"mu": 0.5, "nu": 1.0, "xi": 1.0}              # implementation defaults, not learned
DEFAULT_SR_THRESHOLD = 0.70
DEFAULT_MIN_FONT_SIZE = 16.0
DEFAULT_FILL_THRESHOLD = 0.92   # supplementary estimate_fill threshold only
NON_CONTENT_TITLES = ("source index",)
# Conservative line metrics used ONLY by the Phase 5 PPTX overflow inspector.
# PowerPoint body text typically renders taller than qpp.estimate_fill's 1.2 factor.
_PPTX_LINE_HEIGHT = 1.25
_PPTX_BODY_SPACE_BEFORE = 0.25   # fraction of font size between body bullets when spacing is unset
_PPTX_BULLET_INDENT_PT = 28.0    # approx left indent consumed by a bullet glyph + gap
_PPTX_HEIGHT_TOLERANCE_PT = 1.0  # rounding tolerance when comparing needed vs available height

_MARKER = re.compile(r"\[\s*S(\d+)\s*,\s*p\.\s*(\d+)\s*\]")
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?|(?<![\w.])[-+]?\.\d+")
_TOKEN = r"(?u)\b\w+\b"


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


# =========================================================================================== #
# Dalloc / Dorder / Dstyle / Dpref
# =========================================================================================== #
def dalloc(planned: Mapping[str, int], actual: Mapping[str, int],
           sections: Sequence[str] = CANONICAL_SECTIONS) -> Tuple[float, Dict[str, Any]]:
    rel = [k for k in sections if int(planned.get(k, 0)) > 0 or int(actual.get(k, 0)) > 0]
    terms: Dict[str, Dict[str, float]] = {}
    raw_sum = 0.0
    for k in rel:
        p, a = int(planned.get(k, 0)), int(actual.get(k, 0))
        raw = min(1.0, abs(p - a) / max(1, p))
        raw_sum += raw                                   # sum unrounded; only the reported term is rounded
        terms[k] = {"planned": p, "actual": a, "term": _r(raw)}
    value = _clamp01(raw_sum / len(rel)) if rel else 0.0
    return value, {"formula": "(1/K) * sum_k min(1, |planned_k - actual_k| / max(1, planned_k))",
                   "K": len(rel), "relevant_sections": rel, "terms": terms,
                   "absent_from_both": [k for k in sections if k not in rel]}


def _dedupe(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for s in seq:
        if s not in out:
            out.append(s)
    return out


def dorder(planned_order: Sequence[str], actual_order: Sequence[str]) -> Tuple[float, Dict[str, Any]]:
    a, b = _dedupe(planned_order), _dedupe(actual_order)
    universe = sorted(set(a) | set(b))
    pa, pb = {s: i for i, s in enumerate(a)}, {s: i for i, s in enumerate(b)}
    discordant: List[List[str]] = []
    unmatched: List[List[str]] = []
    for x, y in combinations(universe, 2):
        if x in pa and y in pa and x in pb and y in pb:
            if (pa[x] - pa[y]) * (pb[x] - pb[y]) < 0:
                discordant.append([x, y])
        else:
            unmatched.append([x, y])
    n_pairs = len(universe) * (len(universe) - 1) // 2
    value = _clamp01((len(discordant) + len(unmatched)) / n_pairs) if n_pairs else 0.0
    return value, {"formula": "(discordant pairs + unrankable pairs) / C(|U|, 2)",
                   "planned_order": a, "actual_order": b, "n_pairs": n_pairs,
                   "discordant_pairs": discordant, "unrankable_pairs": unmatched,
                   "missing_from_actual": [s for s in a if s not in pb],
                   "extra_in_actual": [s for s in b if s not in pa]}


def _theta(q: Any) -> Dict[str, Any]:
    """Accept a Phase 1 QPP JSON dict (or QPP object) and return its theta = {w, pi, phi}."""
    if hasattr(q, "to_dict"):
        q = q.to_dict()
    if "theta" in q:
        return q["theta"]
    return q


def dstyle(ref_phi: Mapping[str, float], gen_phi: Mapping[str, float],
           weights: Optional[Mapping[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    w = dict(DEFAULT_STYLE_WEIGHTS if weights is None else weights)
    s = sum(w.values())
    if s <= 0 or any(v < 0 for v in w.values()):
        raise ValueError("style weights must be non-negative and sum to a positive number")
    feats: Dict[str, Dict[str, float]] = {}
    total = 0.0
    for f in ("eta", "beta", "rho", "epsilon"):
        r, g = float(ref_phi.get(f, 0.0)), float(gen_phi.get(f, 0.0))
        d = _clamp01(abs(r - g))
        feats[f] = {"reference": _r(r), "generated": _r(g), "abs_diff": _r(d), "weight": _r(w.get(f, 0.0) / s)}
        total += (w.get(f, 0.0) / s) * d
    return _clamp01(total), {
        "formula": "sum_f weight_f * min(1, |phi_f(reference) - phi_f(generated)|), f in {eta, beta, rho, epsilon}",
        "features_included": ["eta", "beta", "rho", "epsilon"],
        "features_excluded": "colour, font family, animation, layout (not part of the QPP; future work). "
                             "w and pi are covered by Dalloc/Dorder.",
        "features": feats}


def dpref(d_alloc: float, d_order: float, d_style: float,
          weights: Optional[Mapping[str, float]] = None) -> Tuple[float, float, Dict[str, Any]]:
    w = dict(DEFAULT_PREF_WEIGHTS if weights is None else weights)
    if set(w) != {"alloc", "order", "style"} or any(v < 0 for v in w.values()) or abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError("Dpref weights must be {alloc, order, style}, non-negative, and sum to 1")
    val = _clamp01(w["alloc"] * d_alloc + w["order"] * d_order + w["style"] * d_style)
    return val, _clamp01(1.0 - val), {
        "formula": "Dpref = l1*Dalloc + l2*Dorder + l3*Dstyle ; PF = 1 - Dpref",
        "weights": w, "contributions": {"alloc": _r(w["alloc"] * d_alloc), "order": _r(w["order"] * d_order),
                                         "style": _r(w["style"] * d_style)}}


# =========================================================================================== #
# Reading the generated PPTX
# =========================================================================================== #
def _is_non_content(slide) -> bool:
    t = slide.shapes.title
    return t is not None and t.has_text_frame and t.text_frame.text.strip().lower() in NON_CONTENT_TITLES


def read_deck_bullets(pptx_path: Path) -> List[Dict[str, Any]]:
    """Body paragraphs of content slides: text with the trailing source marker split off.

    Content slides = slides with a body placeholder that are not the title-layout slide and not a non-content slide.
    Slide numbers are PPTX slide numbers (1-based)."""
    prs = Presentation(str(pptx_path))
    out: List[Dict[str, Any]] = []
    for n, slide in enumerate(prs.slides, 1):
        if slide.slide_layout.name == "Title Slide" or _is_non_content(slide):
            continue
        title = slide.shapes.title.text_frame.text.strip() if slide.shapes.title is not None else ""
        body = [sh for sh in slide.placeholders if sh.placeholder_format.type in
                (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT) and sh.has_text_frame]
        k = 0
        for sh in body:
            for p in sh.text_frame.paragraphs:
                full = p.text
                if not full.strip():
                    continue
                k += 1
                m = None
                for m in _MARKER.finditer(full):
                    pass                                          # last marker in the paragraph
                text = (full[:m.start()] + full[m.end():]).strip() if m else full.strip()
                out.append({"id": f"s{n}.b{k}", "slide": n, "slide_title": title, "text": text,
                            "sentence_id": int(m.group(1)) if m else None,
                            "page": int(m.group(2)) if m else None,
                            "marker": m.group(0) if m else None})
    return out


# =========================================================================================== #
# Source support ratio (lexical heuristic)
# =========================================================================================== #
def _sentence_index(parsed: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {int(s["id"]): s for s in parsed.get("sentences", [])}


def source_support(bullets: Sequence[Mapping[str, Any]], parsed: Mapping[str, Any],
                   threshold: float = DEFAULT_SR_THRESHOLD) -> Dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("SR threshold must be in [0, 1]")
    sents = _sentence_index(parsed)
    resolved = [b for b in bullets if b.get("sentence_id") in sents]
    sims: Dict[str, float] = {}
    if resolved:
        corpus = [s["text"] for s in sents.values()] + [b["text"] for b in bullets]
        try:
            vec = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN, norm="l2", use_idf=True, smooth_idf=True)
            mat = vec.fit_transform(corpus).tocsr()
        except ValueError:
            mat = None
        if mat is not None:
            row = {sid: i for i, sid in enumerate(sents)}
            n_s = len(sents)
            bidx = {id(b): n_s + i for i, b in enumerate(bullets)}
            for b in resolved:
                v = mat[bidx[id(b)]].multiply(mat[row[b["sentence_id"]]]).sum()
                sims[b["id"]] = float(np.clip(v, 0.0, 1.0))
    per: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []
    for b in bullets:
        rec = {"id": b["id"], "slide": b["slide"], "sentence_id": b.get("sentence_id"), "cosine": None,
               "supported": False, "reason": None}
        sid = b.get("sentence_id")
        if sid is None:
            rec["reason"] = "no source pointer on the slide"
        elif sid not in sents:
            rec["reason"] = f"pointer S{sid:03d} does not exist in the parsed paper"
        elif sents[sid].get("page") != b.get("page"):
            rec["reason"] = f"pointer page p.{b.get('page')} != paper page p.{sents[sid].get('page')}"
            rec["cosine"] = _r(sims.get(b["id"], 0.0))
        else:
            c = sims.get(b["id"], 0.0)
            rec["cosine"] = _r(c)
            rec["supported"] = c >= threshold
            if not rec["supported"]:
                rec["reason"] = f"lexical cosine {c:.3f} < threshold {threshold}"
        per.append(rec)
        if not rec["supported"]:
            unsupported.append({"id": b["id"], "slide": b["slide"], "text": b["text"],
                                "sentence_id": sid, "cosine": rec["cosine"], "reason": rec["reason"]})
    total, sup = len(bullets), sum(1 for r in per if r["supported"])
    return {"method": "lexical TF-IDF cosine between bullet and its source sentence (heuristic; NOT semantic entailment)",
            "threshold": threshold, "supported_bullets": sup, "total_bullets": total,
            "sr": _clamp01(sup / total) if total else 1.0, "vacuous": total == 0,
            "unsupported": unsupported, "unsupported_ids": [u["id"] for u in unsupported], "per_bullet": per}


# =========================================================================================== #
# Numeric fidelity (regex heuristic)
# =========================================================================================== #
def extract_numbers(text: str) -> List[str]:
    """Canonical string for every number token in ``text`` ("1,000" -> "1000", "0.80" -> "0.8")."""
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            d = Decimal(raw)
        except InvalidOperation:
            continue
        out.append(format(d.normalize(), "f") if d != 0 else "0")
    return out


def numeric_fidelity(bullets: Sequence[Mapping[str, Any]], parsed: Mapping[str, Any]) -> Dict[str, Any]:
    sents = _sentence_index(parsed)
    checked = correct = 0
    per: List[Dict[str, Any]] = []
    incorrect: List[Dict[str, Any]] = []
    for b in bullets:
        nums = extract_numbers(b["text"])
        if not nums:
            continue
        src = sents.get(b.get("sentence_id"))
        pool = Counter(extract_numbers(src["text"])) if src else Counter()
        bad, ok = [], []
        for n in nums:
            if pool[n] > 0:
                pool[n] -= 1
                ok.append(n)
            else:
                bad.append(n)
        checked += len(nums)
        correct += len(ok)
        rec = {"id": b["id"], "slide": b["slide"], "sentence_id": b.get("sentence_id"), "numbers": nums,
               "correct": ok, "incorrect": bad, "source_found": src is not None}
        per.append(rec)
        if bad:
            incorrect.append({**rec, "text": b["text"],
                              "source_text": src["text"] if src else None})
    return {"method": "number tokens in the bullet must occur (same value) in its source sentence (regex heuristic)",
            "limitations": ["number words are not parsed", "units / percent signs are ignored",
                            "numbers attached to letters (L2, S000) are not claims",
                            "association of a number with the right quantity is not checked"],
            "numeric_claims_checked": checked, "numeric_claims_correct": correct,
            "nf": _clamp01(correct / checked) if checked else 1.0, "vacuous": checked == 0,
            "bullets_with_numbers": len(per), "incorrect": incorrect, "per_bullet": per}


# =========================================================================================== #
# Overflow rate (inspects the actual generated PPTX)
# =========================================================================================== #
def _shape_inner_box_pt(shape) -> Optional[Tuple[float, float]]:
    """Return (inner_width_pt, inner_height_pt) from the shape's real geometry and text-frame margins."""
    tf = shape.text_frame
    try:
        box_w, box_h = int(shape.width), int(shape.height)
    except Exception:
        return None
    if box_w <= 0 or box_h <= 0:
        return None
    in_w = (box_w - int(tf.margin_left) - int(tf.margin_right)) / 12700.0
    in_h = (box_h - int(tf.margin_top) - int(tf.margin_bottom)) / 12700.0
    if in_w <= 0 or in_h <= 0:
        return None
    return in_w, in_h


def _pptx_needed_height_pt(shape, measurer: TextMeasurer) -> Optional[Dict[str, Any]]:
    """
    Measure how much vertical space the text in ``shape`` needs, using the actual PPTX
    font sizes, paragraph spacing, and shape geometry (python-pptx).

    This is the primary overflow detector.  It is intentionally more conservative than
    qpp.estimate_fill (higher default line height / inter-bullet gap) so that visible
    clipping in PowerPoint is not reported as O = 0.
    """
    inner = _shape_inner_box_pt(shape)
    if inner is None:
        return None
    in_w, in_h = inner
    tf = shape.text_frame
    wrap = tf.word_wrap is not False
    total = 0.0
    n_lines_total = 0
    n_paras = 0
    first = True
    for p in tf.paragraphs:
        text = p.text
        if not text.strip():
            continue
        n_paras += 1
        runs = [r for r in p.runs if r.text.strip()]
        size = max((resolve_font_size(shape, p, r) for r in runs),
                   default=resolve_font_size(shape, p, None))
        has_bullet = paragraph_has_bullet(shape, p)
        indent = (_PPTX_BULLET_INDENT_PT + 18.0 * (p.level or 0)) if has_bullet else 0.0
        avail = max(in_w - indent, size) if wrap else None
        # Count wrapped lines from the actual paragraph text (incl. soft breaks)
        n_lines = 0
        for seg in text.replace("\x0b", "\n").split("\n"):
            n_lines += measurer.wrap_lines(seg if seg else " ", size, avail)
        n_lines = max(1, n_lines)
        n_lines_total += n_lines
        # Line height: honour explicit spacing from the PPTX; otherwise use conservative default
        ls = p.line_spacing
        if ls is None:
            line_h = size * _PPTX_LINE_HEIGHT
        elif isinstance(ls, float):
            line_h = size * _PPTX_LINE_HEIGHT * ls
        else:
            line_h = float(ls.pt)
        if p.space_before is not None:
            before = float(p.space_before.pt)
        else:
            before = (size * _PPTX_BODY_SPACE_BEFORE) if (not first) else 0.0
        after = float(p.space_after.pt) if p.space_after is not None else 0.0
        total += n_lines * line_h + before + after
        first = False
    if total <= 0:
        return None
    return {
        "needed_pt": total,
        "available_pt": in_h,
        "ratio": total / in_h,
        "n_paras": n_paras,
        "n_lines": n_lines_total,
        "inner_width_pt": in_w,
        "overflows": total > in_h + _PPTX_HEIGHT_TOLERANCE_PT,
    }


def overflow_rate(pptx_path: Path, min_font_size: float = DEFAULT_MIN_FONT_SIZE,
                  fill_threshold: float = DEFAULT_FILL_THRESHOLD) -> Dict[str, Any]:
    """
    Inspect every non-empty text shape in the generated PPTX.

    Primary signal (actual PPTX geometry):
        needed text height (from real fonts / spacing / wrap) > text-frame inner height
    Secondary signals:
        any non-marker run below min_font_size
        qpp.estimate_fill > fill_threshold  (supplementary only; cannot clear a primary hit)
    """
    _BODY_TYPES = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT}
    prs = Presentation(str(pptx_path))
    measurer = TextMeasurer()
    boxes: List[Dict[str, Any]] = []
    for n, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            is_body = False
            try:
                if sh.is_placeholder and sh.placeholder_format.type in _BODY_TYPES:
                    is_body = True
            except Exception:
                pass
            if not is_body and "content" in (sh.name or "").lower():
                is_body = True
            measured = _pptx_needed_height_pt(sh, measurer)
            est = estimate_fill(sh, measurer)  # supplementary heuristic only
            small = []
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if not r.text.strip() or _MARKER.fullmatch(r.text.strip()):
                        continue
                    sz = float(resolve_font_size(sh, p, r))
                    if sz < min_font_size:
                        small.append(sz)
            reasons = []
            # Primary: actual PPTX height check.
            # Body/content boxes: any needed > available is overflow (the screenshot failure mode).
            # Titles/subtitles: only flag clear overflow (ratio > 1.05) — template titles are often tight.
            actual_overflow = False
            if measured is not None:
                if is_body and measured["overflows"]:
                    actual_overflow = True
                elif (not is_body) and measured["ratio"] > 1.05:
                    actual_overflow = True
            if actual_overflow:
                reasons.append(
                    f"pptx text height {measured['needed_pt']:.1f} pt > box inner height "
                    f"{measured['available_pt']:.1f} pt "
                    f"(ratio {measured['ratio']:.2f}, {measured['n_lines']} lines / "
                    f"{measured['n_paras']} paras)"
                )
            if small:
                reasons.append(f"font {min(small):g} pt < minimum {min_font_size:g} pt")
            # estimate_fill is supplementary only; cannot clear a primary hit.
            # Use fill_threshold for body boxes; a hard 1.0 ceiling for titles.
            est_thr = fill_threshold if is_body else 1.0
            if est is not None and est > est_thr:
                reasons.append(f"estimate_fill {est:.2f} > {est_thr} (supplementary)")
            boxes.append({
                "slide": n, "shape": sh.name, "box_id": f"slide{n}:{sh.name}",
                "needed_pt": None if not measured else _r(measured["needed_pt"], 2),
                "available_pt": None if not measured else _r(measured["available_pt"], 2),
                "height_ratio": None if not measured else _r(measured["ratio"], 4),
                "n_paras": None if not measured else measured["n_paras"],
                "n_lines": None if not measured else measured["n_lines"],
                "is_body": is_body,
                "estimate_fill": None if est is None else _r(est, 4),
                "min_font_pt": min(small) if small else None,
                "actual_overflow": actual_overflow,
                "overflow": bool(reasons),
                "reasons": reasons,
            })
    bad = [b for b in boxes if b["overflow"]]
    return {
        "method": "primary: actual PPTX text-box geometry (python-pptx shape size, margins, "
                  "run fonts, paragraph spacing, greedy wrap) vs inner height; "
                  "secondary: min font size; supplementary: qpp.estimate_fill",
        "text_measurer": measurer.name,
        "min_font_size": min_font_size,
        "fill_threshold": fill_threshold,
        "line_height_factor": _PPTX_LINE_HEIGHT,
        "body_space_before_factor": _PPTX_BODY_SPACE_BEFORE,
        "total_checked": len(boxes),
        "overflow_count": len(bad),
        "overflow_rate": _clamp01(len(bad) / len(boxes)) if boxes else 0.0,
        "affected_slides": sorted({b["slide"] for b in bad}),
        "affected_boxes": [b["box_id"] for b in bad],
        "boxes": boxes,
    }


def overflow_from_counts(overflowed: int, total: int) -> float:
    """O = overflowed / total (0 when nothing was checked)."""
    return _clamp01(overflowed / total) if total else 0.0


def objective(d_pref: float, o: float, nf: float, sr: float,
              weights: Optional[Mapping[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
    w = dict(DEFAULT_OBJECTIVE_WEIGHTS if weights is None else weights)
    if set(w) != {"mu", "nu", "xi"} or any(v < 0 for v in w.values()):
        raise ValueError("objective weights must be {mu, nu, xi}, non-negative")
    comp = {"dpref": d_pref, "mu_O": w["mu"] * o, "nu_1_minus_NF": w["nu"] * (1.0 - nf),
            "xi_1_minus_SR": w["xi"] * (1.0 - sr)}
    j = sum(comp.values())
    return float(j), {"formula": "J(S) = Dpref + mu*O + nu*(1 - NF) + xi*(1 - SR)   (lower is better)",
                      "weights": w, "components": {k: _r(v) for k, v in comp.items()}}


# =========================================================================================== #
# Generated-deck QPP (Phase 1 extractor on a copy without non-content slides)
# =========================================================================================== #
def generated_qpp(pptx_path: Path) -> Tuple[Dict[str, Any], List[int]]:
    prs = Presentation(str(pptx_path))
    drop = [i for i, s in enumerate(prs.slides) if _is_non_content(s)]
    if not drop:
        return extract_qpp(pptx_path).to_dict(), []
    lst = prs.slides._sldIdLst
    ids = list(lst)
    for i in drop:
        prs.part.drop_rel(ids[i].rId)
        lst.remove(ids[i])
    with tempfile.TemporaryDirectory(prefix="at_verify_") as tmp:
        p = Path(tmp) / "generated_content_only.pptx"
        prs.save(str(p))
        d = extract_qpp(p).to_dict()
    d["source"] = str(pptx_path)
    return d, [i + 1 for i in drop]


def _tv(w1: Mapping[str, float], w2: Mapping[str, float]) -> float:
    return _clamp01(0.5 * sum(abs(float(w1.get(k, 0.0)) - float(w2.get(k, 0.0))) for k in CANONICAL_SECTIONS))


# =========================================================================================== #
# Top level
# =========================================================================================== #
def verify(parsed: Mapping[str, Any], ref_qpp: Any, plan: Mapping[str, Any], draft: Mapping[str, Any],
           pptx_path: Path, *, pref_weights: Optional[Mapping[str, float]] = None,
           style_weights: Optional[Mapping[str, float]] = None,
           objective_weights: Optional[Mapping[str, float]] = None,
           sr_threshold: float = DEFAULT_SR_THRESHOLD, min_font_size: float = DEFAULT_MIN_FONT_SIZE,
           fill_threshold: float = DEFAULT_FILL_THRESHOLD,
           inputs: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    pptx_path = Path(pptx_path)
    if not pptx_path.is_file():
        raise FileNotFoundError(f"generated PPTX not found: {pptx_path}")
    warnings: List[str] = []

    gen_q, dropped = generated_qpp(pptx_path)
    ref_t, gen_t = _theta(ref_qpp), _theta(gen_q)
    planned = {k: int(v) for k, v in (plan.get("slide_allocation") or {}).items()}
    planned_order = list(plan.get("section_order") or plan.get("presented_section_order") or [])
    actual = {k: int(v) for k, v in (gen_q.get("section_slide_counts") or {}).items()}

    d_alloc, alloc_d = dalloc(planned, actual)
    d_order, order_d = dorder(planned_order, gen_t.get("pi", []))
    d_style, style_d = dstyle(ref_t["phi"], gen_t["phi"], style_weights)
    style_d["diagnostic_w_total_variation_vs_reference"] = _r(_tv(ref_t["w"], gen_t["w"]))
    style_d["diagnostic_pi_kendall_vs_reference"] = _r(dorder(ref_t.get("pi", []), gen_t.get("pi", []))[0])
    d_pref, pf, pref_d = dpref(d_alloc, d_order, d_style, pref_weights)

    bullets = read_deck_bullets(pptx_path)
    sr_d = source_support(bullets, parsed, sr_threshold)
    nf_d = numeric_fidelity(bullets, parsed)
    ov_d = overflow_rate(pptx_path, min_font_size, fill_threshold)
    o = overflow_from_counts(ov_d["overflow_count"], ov_d["total_checked"])
    j, obj_d = objective(d_pref, o, nf_d["nf"], sr_d["sr"], objective_weights)

    # cross-checks against the Phase 3 draft (informational; never changes a metric)
    d_total = sum(len(s.get("bullets", [])) for s in draft.get("slides", []))
    d_ptrs = {int(b["source"]["sentence_id"][1:]) for s in draft.get("slides", []) for b in s.get("bullets", [])}
    deck_ptrs = {b["sentence_id"] for b in bullets if b["sentence_id"] is not None}
    cross = {"draft_bullets": d_total, "deck_bullets": len(bullets),
             "pointers_only_in_deck": sorted(deck_ptrs - d_ptrs), "pointers_only_in_draft": sorted(d_ptrs - deck_ptrs)}
    if d_total != len(bullets):
        warnings.append(f"deck has {len(bullets)} bullets but the Phase 3 draft has {d_total}")
    if cross["pointers_only_in_deck"] or cross["pointers_only_in_draft"]:
        warnings.append("source pointers in the deck differ from the Phase 3 draft")
    if dropped:
        warnings.append(f"non-content slide(s) {dropped} excluded from the generated-deck QPP")
    if sr_d["vacuous"]:
        warnings.append("deck has no bullets: SR = 1 and NF = 1 are vacuous")
    if gen_q.get("n_content_slides", 0) == 0:
        warnings.append("generated deck has no content slides: Dalloc/Dorder/Dstyle describe an empty deck")
    for w in gen_q.get("warnings", []):
        warnings.append(f"generated-deck QPP: {w}")
    warnings.extend(f"reference QPP: {w}" for w in (ref_qpp.get("warnings", []) if isinstance(ref_qpp, dict) else []))
    if sr_d["unsupported"]:
        warnings.append(f"{len(sr_d['unsupported'])} bullet(s) failed the lexical source-support heuristic")
    if nf_d["incorrect"]:
        warnings.append(f"{len(nf_d['incorrect'])} bullet(s) contain numbers not found in their source sentence")
    if ov_d["overflow_count"]:
        warnings.append(f"{ov_d['overflow_count']} text box(es) overflow or violate the minimum font size")

    return {
        "schema": SCHEMA,
        "mode": "deterministic verification (lexical / regex / geometry heuristics; no LLM, no API)",
        "inputs": {"pptx": str(pptx_path), **(inputs or {})},
        "dalloc": _r(d_alloc), "dorder": _r(d_order), "dstyle": _r(d_style),
        "dpref": _r(d_pref), "pf": _r(pf),
        "sr": _r(sr_d["sr"]), "nf": _r(nf_d["nf"]), "overflow_rate": _r(o),
        "objective": _r(j),
        "config": {"pref_weights": pref_d["weights"], "style_weights": dict(style_weights or DEFAULT_STYLE_WEIGHTS),
                   "objective_weights": obj_d["weights"], "sr_threshold": sr_threshold,
                   "min_font_size": min_font_size, "fill_threshold": fill_threshold},
        "details": {
            "dalloc": alloc_d, "dorder": order_d, "dstyle": style_d, "dpref": pref_d,
            "sr": sr_d, "nf": nf_d, "overflow": ov_d, "objective": obj_d,
            "generated_qpp": {"w": gen_t["w"], "pi": gen_t["pi"], "phi": gen_t["phi"],
                              "section_slide_counts": gen_q.get("section_slide_counts"),
                              "n_content_slides": gen_q.get("n_content_slides"), "excluded_slides": dropped},
            "reference_qpp": {"w": ref_t["w"], "pi": ref_t["pi"], "phi": ref_t["phi"]},
            "plan": {"slide_allocation": planned, "section_order": planned_order},
            "draft_cross_check": cross,
        },
        "warnings": warnings,
    }

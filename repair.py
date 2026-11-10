"""
repair.py -- AuditTailor Phase 6: bounded, deterministic, extractive repair loop.

    verify -> (criteria met?) -> repair -> verify again -> keep the candidate with the LOWEST J(S)

No LLM, no API, no network, no randomness.  Nothing is invented: every repair only (a) re-fits text that already
exists, (b) restores text from the Phase 3 draft / the parsed source sentence it points to, or (c) restores the
structure the Phase 2 plan prescribes.  The reference QPP and the plan are never modified.

Repair priority (proposal): 1. overflow   2. support / numeric fidelity   3. preference drift

    overflow    (1) reuses Phase 4's fitter (pptx_generator.fit_body: font steps down to the minimum, then tighter
                spacing); runs below the minimum font are raised to it first.  (2) If still overflowing after
                reasonable font/spacing adjustment, split trailing bullets onto a continuation slide that uses the
                same default content-slide layout/style.  Source pointers stay with their bullets; no bullets are
                deleted, rewritten, or numerically altered.  Continuation is the preferred solution for genuinely
                dense slides.
    support     each failing bullet is replaced by the first candidate that passes the verifier's own checks:
                (1) the Phase 3 draft text of that bullet, (2) its original (untrimmed) text, (3) the parsed source
                sentence it points to.  The source pointer is preserved / restored.  Numbers are never "guessed":
                they come back with the source-grounded text.
    preference  structural drift from the Phase 2 plan: slide titles (section labels), slide order and allocation.
                One-to-one slide matches are retitled / reordered in place; missing, extra or duplicated slides
                trigger a rebuild with Phase 4's generator from the plan + draft.  Dstyle is NOT repairable here
                (it would require changing content), and is reported as such.

Round structure: one round = every failing family, in priority order, followed by an overflow re-fit if content
was restored.  The round's candidate is verified; it becomes the best candidate only if J strictly decreases.
If it does not improve, the loop returns to the best candidate and blocks the families that were tried; it stops when
all criteria hold, when no unblocked failing family remains, when a round changes nothing, or after MAX_REPAIR_ROUNDS.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER

import pptx_generator as G
import verifier as V
from qpp import TextMeasurer, resolve_font_size

SCHEMA = "audittailor.repair_history.v1"

# =========================================================================================== #
# Configuration -- the ONE place to change repair behaviour / thresholds                       #
# =========================================================================================== #
MAX_REPAIR_ROUNDS = 3          # hard bound on repair rounds (each round repairs every failing family)
MAX_OVERFLOW_RATE = 0.0        # O must be <= this
MIN_SUPPORT_RATIO = 1.0        # SR must be >= this
MIN_NUMERIC_FIDELITY = 1.0     # NF must be >= this
MAX_PREFERENCE_DISTANCE = 0.15  # Dpref must be <= this (implementation default; clean decks sit near 0.09, all Dstyle)
MAX_DALLOC = 0.0               # the plan is authoritative for allocation ...
MAX_DORDER = 0.0               # ... and for section order, so any drift is repairable and is repaired

FAMILIES = ("overflow", "support", "preference")          # repair priority, highest first


@dataclass(frozen=True)
class RepairConfig:
    max_rounds: int = MAX_REPAIR_ROUNDS
    max_overflow_rate: float = MAX_OVERFLOW_RATE
    min_support_ratio: float = MIN_SUPPORT_RATIO
    min_numeric_fidelity: float = MIN_NUMERIC_FIDELITY
    max_preference_distance: float = MAX_PREFERENCE_DISTANCE
    max_dalloc: float = MAX_DALLOC
    max_dorder: float = MAX_DORDER

    def __post_init__(self):
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")
        for name in ("max_overflow_rate", "min_support_ratio", "min_numeric_fidelity", "max_preference_distance",
                     "max_dalloc", "max_dorder"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


# =========================================================================================== #
# Small helpers                                                                                #
# =========================================================================================== #
def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _sid(s: str) -> int:
    return int(str(s).lstrip("S"))


def _metrics(v: Mapping[str, Any]) -> Dict[str, float]:
    return {k: v[k] for k in ("dalloc", "dorder", "dstyle", "dpref", "pf", "sr", "nf", "overflow_rate", "objective")}


def detect_failures(v: Mapping[str, Any], cfg: RepairConfig) -> List[Dict[str, Any]]:
    """Failed criteria of a verification result, in repair-priority order."""
    out: List[Dict[str, Any]] = []

    def add(family, metric, value, op, thr):
        out.append({"family": family, "metric": metric, "value": value, "threshold": thr, "criterion": f"{metric} {op} {thr}"})
    if v["overflow_rate"] > cfg.max_overflow_rate:
        add("overflow", "overflow_rate", v["overflow_rate"], "<=", cfg.max_overflow_rate)
    if v["sr"] < cfg.min_support_ratio:
        add("support", "sr", v["sr"], ">=", cfg.min_support_ratio)
    if v["nf"] < cfg.min_numeric_fidelity:
        add("support", "nf", v["nf"], ">=", cfg.min_numeric_fidelity)
    if v["dalloc"] > cfg.max_dalloc:
        add("preference", "dalloc", v["dalloc"], "<=", cfg.max_dalloc)
    if v["dorder"] > cfg.max_dorder:
        add("preference", "dorder", v["dorder"], "<=", cfg.max_dorder)
    if v["dpref"] > cfg.max_preference_distance:
        add("preference", "dpref", v["dpref"], "<=", cfg.max_preference_distance)
    return out


def _save(prs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(path))
    G._normalise_zip_timestamps(path)           # byte-reproducible output


def _xml(el) -> bytes:
    return etree.tostring(el._element if hasattr(el, "_element") else el)


def _body_shapes(slide):
    return [sh for sh in slide.placeholders
            if sh.placeholder_format.type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT) and sh.has_text_frame]


def _body_paragraphs(slide) -> List[Any]:
    """Non-empty body paragraphs in the same order as verifier.read_deck_bullets (id s<slide>.b<k>, k = 1-based)."""
    return [p for sh in _body_shapes(slide) for p in sh.text_frame.paragraphs if p.text.strip()]


def _content_slide_numbers(prs) -> List[int]:
    """1-based numbers of content slides, by the verifier's own rule."""
    return [n for n, s in enumerate(prs.slides, 1) if s.slide_layout.name != "Title Slide" and not V._is_non_content(s)]


def _marker_run(p):
    runs = list(p.runs)
    return runs[-1] if runs and V._MARKER.search(runs[-1].text) and len(runs) > 1 else None


# =========================================================================================== #
# Family 1: overflow                                                                           #
# =========================================================================================== #
def _fit_shape(sh, min_font: float) -> Dict[str, Any]:
    """Raise sub-minimum runs, then reuse Phase 4's fit_body.  Returns a record (also ``changed``)."""
    before = _xml(sh)
    sizes = [(p, r, float(resolve_font_size(sh, p, r))) for p in sh.text_frame.paragraphs for r in p.runs
             if r.text.strip() and not V._MARKER.fullmatch(r.text.strip())]
    ok = [s for (_p, _r_, s) in sizes if s >= min_font]
    raise_to = max(ok) if ok else float(min_font)
    raised = 0
    for p, r, s in sizes:
        if s < min_font:
            r.font.size = G.Pt(raise_to)
            raised += 1
    pars = []
    for p in sh.text_frame.paragraphs:
        if not p.text.strip():
            continue
        text_runs = [r for r in p.runs if not V._MARKER.fullmatch(r.text.strip())]
        mr = _marker_run(p)
        if text_runs:
            pars.append((p, text_runs[0], mr))
    rec = G.fit_body(sh, pars, TextMeasurer(), min_font, None)
    for p, r0, _mr in pars:                                 # keep multi-run paragraphs at one size
        if r0.font.size is not None:
            for r in p.runs:
                if r._r is not r0._r and not V._MARKER.fullmatch(r.text.strip()) and r.text.strip():
                    r.font.size = r0.font.size
    rec["runs_raised_to_minimum"] = raised
    rec["changed"] = _xml(sh) != before
    return rec


def _paragraph_parts(p) -> Tuple[str, str]:
    """Split a body paragraph into (main_text, marker_suffix). Marker is preserved verbatim."""
    runs = list(p.runs)
    if not runs:
        return "", ""
    mr = _marker_run(p)
    if mr is not None:
        main = "".join(r.text for r in runs if r._r is not mr._r)
        return main.rstrip(), mr.text
    # marker may be inline in the last run
    full = p.text
    m = V._MARKER.search(full)
    if m:
        return full[:m.start()].rstrip(), full[m.start():]
    return full, ""


def _set_paragraph_text(p, main: str, marker_suffix: str) -> None:
    """Rewrite paragraph runs: main text + optional source-marker run. Preserves level."""
    # Clear existing runs
    for r in list(p.runs):
        r.text = ""
    runs = list(p.runs)
    if runs:
        runs[0].text = main
        for r in runs[1:]:
            r.text = ""
    else:
        p.add_run().text = main
    if marker_suffix:
        # Prefer a dedicated trailing run so marker stays styled small/grey when re-fit
        mr = _marker_run(p)
        if mr is not None:
            mr.text = marker_suffix if marker_suffix.startswith(" ") else ("  " + marker_suffix.lstrip())
        else:
            nr = p.add_run()
            nr.text = marker_suffix if marker_suffix.startswith(" ") else ("  " + marker_suffix.lstrip())


def _shorten_main(main: str, target_chars: int) -> str:
    """Trim main text to ~target_chars at a word boundary. Never strips digits/percent tokens mid-token."""
    main = main.rstrip()
    if len(main) <= target_chars or target_chars < 12:
        return main
    cut = main[:target_chars].rstrip()
    # Prefer last whitespace so we don't leave a dangling partial word
    sp = cut.rfind(" ")
    if sp >= max(8, target_chars // 3):
        cut = cut[:sp]
    return cut.rstrip(" ,;:.-") + "…"


def _still_overflows(sh, min_font: float) -> bool:
    """True when Phase 5's actual-PPTX height check still reports overflow."""
    measured = V._pptx_needed_height_pt(sh, TextMeasurer())
    if measured is not None and measured["overflows"]:
        return True
    # Supplementary: estimate_fill above threshold
    from qpp import estimate_fill
    est = estimate_fill(sh, TextMeasurer())
    return est is not None and est > V.DEFAULT_FILL_THRESHOLD


def _shorten_bullets_to_fit(sh, min_font: float) -> Dict[str, Any]:
    """
    Progressively shorten the longest body bullets (word-boundary trim, marker kept)
    until the box fits or no further safe trim is possible. Numbers and source markers
    are never dropped; only prose after them may be truncated.
    """
    shortened: List[Dict[str, Any]] = []
    for _round in range(40):  # hard bound
        if not _still_overflows(sh, min_font):
            break
        # Rank non-empty paragraphs by main-text length (longest first)
        cands = []
        for p in sh.text_frame.paragraphs:
            if not p.text.strip():
                continue
            main, marker = _paragraph_parts(p)
            if len(main) < 16:
                continue
            cands.append((len(main), p, main, marker))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        _, p, main, marker = cands[0]
        # Aggressive trim: 25% per step once text is long, else 15%
        frac = 0.75 if len(main) > 80 else 0.85
        target = max(12, int(len(main) * frac))
        new_main = _shorten_main(main, target)
        if new_main == main or len(new_main) >= len(main):
            break
        _set_paragraph_text(p, new_main, marker)
        shortened.append({"before_len": len(main), "after_len": len(new_main), "marker_kept": bool(marker)})
        # Re-apply font/spacing fit after each trim so estimate stays consistent
        _fit_shape(sh, min_font)
    return {"shortened_count": len(shortened), "shortened": shortened,
            "resolved": not _still_overflows(sh, min_font)}


def _extract_bullet_items(sh) -> List[Dict[str, Any]]:
    """Body paragraphs as {text, marker, level} for re-filling."""
    items = []
    for p in sh.text_frame.paragraphs:
        if not p.text.strip():
            continue
        main, marker = _paragraph_parts(p)
        items.append({"text": main, "marker": marker.strip(), "level": int(p.level or 0)})
    return items


def _continuation_title(base_title: str) -> str:
    """Build the next continuation title: 'X — continued', then 'X — continued 2', etc.

    Titles are kept short enough to fit a single line in typical template title boxes.
    """
    base = base_title.strip()
    m = re.search(r"^(.*?)(?:\s*—\s*continued(?:\s+(\d+))?)$", base, re.IGNORECASE)
    if m:
        stem = m.group(1).rstrip()
        n = int(m.group(2) or 1) + 1
        suffix = f" — continued {n}"
    else:
        stem = base
        suffix = " — continued"
    head = stem.split(":")[0].strip()
    head = re.sub(r"\s*\(\d+/\d+\)\s*$", "", head).strip() or stem
    if len(head) + len(suffix) <= 48:
        stem = head
    elif len(stem) + len(suffix) > 48:
        stem = head[: max(1, 48 - len(suffix))].rstrip()
    return f"{stem}{suffix}"


def _split_overflow_to_continuation(prs, slide_idx: int, sh, min_font: float) -> Dict[str, Any]:
    """
    Move trailing bullets from an overflowing body box onto a new continuation slide
    until the original box fits. Source markers and bullet text are preserved exactly.
    Uses the same default content-slide layout/style. Does not delete, rewrite, or
    numerically alter any bullet. If the continuation itself still overflows, it is
    split further (recursive).
    """
    items = _extract_bullet_items(sh)
    if len(items) <= 1:
        return {"split": False, "moved": 0, "reason": "single bullet cannot be split"}

    # Binary-ish search: keep prefix that fits, move the rest
    keep = len(items)
    moved_items: List[Dict[str, Any]] = []
    for k in range(len(items) - 1, 0, -1):
        # Rebuild shape with first k items
        keep_items = items[:k]
        G._fill_body(sh, [{"text": it["text"], "marker": it["marker"] or None, "level": it["level"]}
                          for it in keep_items])
        # Clear any leftover empty paragraphs beyond what _fill_body wrote
        tf = sh.text_frame
        # _fill_body reuses paragraphs[0] and add_paragraph; excess old paras may remain — strip them
        while len(tf.paragraphs) > len(keep_items):
            p_el = tf.paragraphs[-1]._p
            p_el.getparent().remove(p_el)
        _fit_shape(sh, min_font)
        if not _still_overflows(sh, min_font):
            keep = k
            moved_items = items[k:]
            break
    else:
        # Even one bullet still overflows: keep 1, move the rest (do not delete the long bullet)
        keep = 1
        keep_items = items[:1]
        G._fill_body(sh, [{"text": it["text"], "marker": it["marker"] or None, "level": it["level"]}
                          for it in keep_items])
        while len(sh.text_frame.paragraphs) > 1:
            p_el = sh.text_frame.paragraphs[-1]._p
            p_el.getparent().remove(p_el)
        _fit_shape(sh, min_font)
        moved_items = items[1:]

    if not moved_items:
        return {"split": False, "moved": 0, "reason": "prefix already fits"}

    # Title of the source slide (for continuation naming)
    src_slide = prs.slides[slide_idx]
    title_ph = next((p for p in src_slide.placeholders
                     if p.placeholder_format.type in (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)), None)
    base_title = title_ph.text_frame.text.strip() if title_ph is not None else "Continued"
    cont_title = _continuation_title(base_title)

    # Insert continuation slide immediately after the overflowing one
    layout = src_slide.slide_layout
    # python-pptx has no public insert; append then reorder XML
    new_slide = prs.slides.add_slide(layout)
    # Move the new slide element to sit right after slide_idx
    sldIdLst = prs.slides._sldIdLst
    new_id = sldIdLst[-1]
    sldIdLst.remove(new_id)
    sldIdLst.insert(slide_idx + 1, new_id)

    tph = next((p for p in new_slide.placeholders
                if p.placeholder_format.type in (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)), None)
    if tph is not None:
        tph.text_frame.text = cont_title
    body = next((p for p in new_slide.placeholders
                 if p.placeholder_format.type in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT)), None)
    further_splits: List[Dict[str, Any]] = []
    if body is not None:
        pars = G._fill_body(body, [{"text": it["text"], "marker": it["marker"] or None, "level": it["level"]}
                                   for it in moved_items])
        G.fit_body(body, pars, TextMeasurer(), min_font, None)
        # If continuation itself still overflows, prefer another continuation split (no shortening)
        if _still_overflows(body, min_font):
            _fit_shape(body, min_font)
            if _still_overflows(body, min_font) and len(moved_items) > 1:
                further = _split_overflow_to_continuation(prs, slide_idx + 1, body, min_font)
                further_splits.append(further)

    # Notes: preserve pointers
    try:
        lines = [f"Continuation of: {base_title}", "Source pointers:"]
        for i, it in enumerate(moved_items, 1):
            lines.append(f"{i}. {it['marker'] or '(no marker)'} — {it['text'][:80]}")
        G._set_notes(new_slide, lines)
    except Exception:
        pass

    return {"split": True, "moved": len(moved_items), "kept": keep,
            "continuation_title": cont_title, "markers_moved": [it["marker"] for it in moved_items],
            "further_splits": further_splits}


def repair_overflow(src: Path, dst: Path, boxes: Sequence[Mapping[str, Any]], min_font: float) -> Dict[str, Any]:
    """
    Repair overflowing text boxes, in priority order:
      1. Raise sub-minimum fonts, reduce font size, tighten spacing (Phase 4 fitter — reasonable only).
      2. If still overflowing after reasonable fit: split trailing bullets onto a continuation slide
         (same default content-slide layout/style). Continuation is preferred for genuinely dense slides.
    Source pointers remain attached to their bullets; no bullets are deleted or rewritten; numbers and
    citations are preserved exactly; the PPTX stays fully editable. Continuation slides are created
    before any text is allowed to remain outside the content area.
    """
    prs = Presentation(str(src))
    fixes: List[Dict[str, Any]] = []
    # Process higher slide numbers first so newly inserted continuation slides do not
    # shift the indices of remaining boxes within this call.
    ordered = sorted(boxes, key=lambda b: -int(b["slide"]))
    for b in ordered:
        slide_n = int(b["slide"])
        if slide_n < 1 or slide_n > len(prs.slides):
            continue
        slide = prs.slides[slide_n - 1]
        sh = next((s for s in slide.shapes if s.name == b["shape"] and s.has_text_frame and s.text_frame.text.strip()), None)
        if sh is None:
            continue
        rec = _fit_shape(sh, min_font)
        entry: Dict[str, Any] = {
            "slide": slide_n, "shape": b["shape"], "changed": rec["changed"],
            "font_before_pt": rec["font_size_start_pt"], "font_after_pt": rec["font_size_final_pt"],
            "runs_raised_to_minimum": rec["runs_raised_to_minimum"], "spacing_tightened": rec["spacing_tightened"],
            "fill_before": rec["fill_before"], "fill_after": rec["fill_after"],
            "resolved_by_estimate": rec["overflow_resolved"],
            "shortened": False, "split": False,
        }
        # After reasonable font/spacing adjustment, prefer continuation over any text shortening.
        # Also prefer continuation for genuinely dense body boxes (many bullets) where the
        # estimate claims a fit only after aggressive font reduction.
        n_paras = len([p for p in sh.text_frame.paragraphs if (p.text or "").strip()])
        dense_body = n_paras >= 8 and float(rec.get("fill_before") or 0) > 1.2
        still = _still_overflows(sh, min_font)
        # Marginal residual overflow after min-font fit on a short box (1-3 paras)
        # is left for support/NF repair rather than forced continuation.
        marginal = (
            not dense_body
            and n_paras <= 3
            and float(rec.get("fill_after") or 0) < 1.15
            and float(rec.get("font_size_final_pt") or 0) <= float(min_font) + 0.01
        )
        needs_split = ((not entry["resolved_by_estimate"] or still) or dense_body) and not marginal
        if needs_split:
            if still or dense_body:
                split_rec = _split_overflow_to_continuation(prs, slide_n - 1, sh, min_font)
                entry["split"] = bool(split_rec.get("split"))
                entry["split_detail"] = split_rec
                entry["changed"] = entry["changed"] or entry["split"]
                entry["resolved_by_estimate"] = not _still_overflows(sh, min_font)
        fixes.append(entry)
    changed = [f for f in fixes if f["changed"]]
    if changed:
        _save(prs, dst)
    return {"changed": bool(changed), "boxes": fixes, "affected_slides": sorted({f["slide"] for f in changed})}


# =========================================================================================== #
# Slide / bullet matching against the Phase 3 draft                                           #
# =========================================================================================== #
def _draft_slides(draft: Mapping[str, Any], plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Draft slides that carry bullets, in the plan's order (the generator skips empty slides)."""
    by = {(s["section"], s["section_slide_index"]): s for s in draft["slides"]}
    out = []
    for e in G.expected_content_slides(plan):
        d = by.get((e["section"], e["section_slide_index"]))
        if d is not None and d["bullets"]:
            out.append({**e, "draft": d})
    return out


def _match_slides(deck_bullets: Sequence[Mapping[str, Any]], content_nums: Sequence[int],
                  expected: Sequence[Mapping[str, Any]]) -> Dict[int, Optional[int]]:
    """deck slide number -> index into ``expected`` (or None).  By source-pointer overlap, ties -> earliest in the plan;
    leftover deck slides are matched by position only when the deck has exactly the planned number of slides."""
    ptr_owner: Dict[int, int] = {}
    for i, e in enumerate(expected):
        for b in e["draft"]["bullets"]:
            ptr_owner.setdefault(_sid(b["source"]["sentence_id"]), i)
    match: Dict[int, Optional[int]] = {}
    for n in content_nums:
        votes: Dict[int, int] = {}
        for b in deck_bullets:
            if b["slide"] == n and b["sentence_id"] in ptr_owner:
                votes[ptr_owner[b["sentence_id"]]] = votes.get(ptr_owner[b["sentence_id"]], 0) + 1
        match[n] = min(votes, key=lambda i: (-votes[i], i)) if votes else None
    if len(content_nums) == len(expected):
        used = {i for i in match.values() if i is not None}
        for pos, n in enumerate(content_nums):
            if match[n] is None and pos not in used:
                match[n], _ = pos, used.add(pos)
    return match


# =========================================================================================== #
# Family 2: source support / numeric fidelity                                                  #
# =========================================================================================== #
def _grounded(text: str, sid: int, page: int, parsed: Mapping[str, Any], threshold: float) -> bool:
    """Would this bullet pass the verifier's SR and NF checks on its own?"""
    b = {"id": "candidate", "slide": 0, "text": text, "sentence_id": sid, "page": page}
    return (V.source_support([b], parsed, threshold)["supported_bullets"] == 1
            and V.numeric_fidelity([b], parsed)["nf"] == 1.0)


def _candidates(draft_bullet: Optional[Mapping[str, Any]], deck_sid: Optional[int], parsed: Mapping[str, Any]
                ) -> Tuple[List[Tuple[str, str]], Optional[int], Optional[int]]:
    """Ordered (label, text) replacement candidates + the source pointer (sentence id, page) they are grounded in."""
    sents = V._sentence_index(parsed)
    cands: List[Tuple[str, str]] = []
    if draft_bullet is not None:
        sid, page = _sid(draft_bullet["source"]["sentence_id"]), int(draft_bullet["source"]["page"])
        cands += [("phase3_draft_text", draft_bullet["text"])]
        cands.append(("phase3_original_sentence", draft_bullet.get("original_text") or ""))
    elif deck_sid in sents:
        sid, page = deck_sid, int(sents[deck_sid]["page"])
    else:
        return [], None, None
    if sid in sents:
        cands.append(("parsed_source_sentence", sents[sid]["text"]))
    seen, out = set(), []
    for lab, t in cands:
        if t.strip() and t not in seen:
            seen.add(t)
            out.append((lab, t))
    return out, sid, page


def _write_bullet(p, text: str, marker: str) -> None:
    """Replace a paragraph's text and source marker, keeping the first run's formatting."""
    runs = list(p.runs)
    if not runs:
        p.add_run()
        runs = list(p.runs)
    first = runs[0]
    mr = _marker_run(p)
    first.text = text
    for r in runs[1:]:
        if mr is None or r._r is not mr._r:
            r._r.getparent().remove(r._r)
    if mr is None:
        mr = p.add_run()
        G._set_marker_style(mr, float(first.font.size.pt) if first.font.size is not None else 20.0)
    mr.text = "  " + marker


def repair_support(src: Path, dst: Path, ver: Mapping[str, Any], parsed: Mapping[str, Any],
                   draft: Dict[str, Any], plan: Mapping[str, Any], threshold: float) -> Dict[str, Any]:
    """Restore failing bullets (SR / NF) from the Phase 3 draft or the parsed source sentence.  Mutates ``draft`` only
    when the draft's own bullet fails verification and a deeper source-grounded candidate has to be used."""
    prs = Presentation(str(src))
    bullets = V.read_deck_bullets(src)
    by_id = {b["id"]: b for b in bullets}
    failing: Dict[str, List[str]] = {}
    for u in ver["details"]["sr"]["unsupported"]:
        failing.setdefault(u["id"], []).append("sr")
    for u in ver["details"]["nf"]["incorrect"]:
        failing.setdefault(u["id"], []).append("nf")
    expected = _draft_slides(draft, plan)
    match = _match_slides(bullets, _content_slide_numbers(prs), expected)
    done: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for bid in sorted(failing, key=lambda s: (int(s[1:].split(".")[0]), int(s.split(".b")[1]))):
        b = by_id.get(bid)
        if b is None:
            continue
        n, k = b["slide"], int(bid.split(".b")[1])
        ei = match.get(n)
        draft_bullet = None
        if ei is not None:
            dbs = expected[ei]["draft"]["bullets"]
            if k - 1 < len(dbs):
                draft_bullet = dbs[k - 1]
            else:
                draft_bullet = next((x for x in dbs if _sid(x["source"]["sentence_id"]) == b["sentence_id"]), None)
        cands, sid, page = _candidates(draft_bullet, b["sentence_id"], parsed)
        chosen = next(((lab, t) for lab, t in cands if _grounded(t, sid, page, parsed, threshold)), None)
        marker = f"[S{sid:03d}, p.{page}]" if sid is not None else None
        if chosen is None or (chosen[1] == b["text"] and b["marker"] == marker):
            skipped.append({"id": bid, "why": failing[bid],
                            "reason": "no source-grounded replacement available" if chosen is None
                            else "bullet already equals its best source-grounded candidate"})
            continue
        paras = _body_paragraphs(prs.slides[n - 1])
        _write_bullet(paras[k - 1], chosen[1], marker)
        if draft_bullet is not None and chosen[0] != "phase3_draft_text":       # keep draft consistent with the deck
            draft_bullet["text"] = chosen[1]
            draft_bullet.setdefault("trim_ops", []).append(f"repair:{chosen[0]}")
        done.append({"id": bid, "slide": n, "why": failing[bid], "restored_from": chosen[0],
                     "source_pointer": {"sentence_id": f"S{sid:03d}", "page": page},
                     "old_text": b["text"], "new_text": chosen[1]})
    if done:
        _save(prs, dst)
    return {"changed": bool(done), "bullets": done, "unrepaired": skipped,
            "affected_slides": sorted({d["slide"] for d in done}), "affected_bullets": [d["id"] for d in done]}


# =========================================================================================== #
# Family 3: preference drift (allocation / order / section labels)                             #
# =========================================================================================== #
def _planned_title(e: Mapping[str, Any]) -> str:
    return e["section"] + (f" ({e['section_slide_index']}/{e['section_slide_count']})" if e["section_slide_count"] > 1 else "")


def _refresh_index_and_notes(prs, placed: Sequence[Tuple[int, Mapping[str, Any]]]) -> None:
    """After a reorder/retitle: bring the Source Index slide and the 'Section:' note line back in line with the plan."""
    for n, e in placed:
        s = prs.slides[n - 1]
        if s.has_notes_slide:
            ps = s.notes_slide.notes_text_frame.paragraphs
            if ps and ps[0].text.startswith("Section:") and ps[0].runs:
                ps[0].runs[0].text = f"Section: {e['section']}"
                for r in ps[0].runs[1:]:
                    r.text = ""
    for s in prs.slides:
        if V._is_non_content(s):
            paras = [p for sh in _body_shapes(s) for p in sh.text_frame.paragraphs if p.text.strip()]
            if len(paras) == len(placed):
                for p, (n, e) in zip(paras, placed):
                    markers = [G.source_marker(b["source"]) for b in e["draft"]["bullets"]]
                    if p.runs:
                        p.runs[0].text = f"Slide {n} · {e['section']}: " + ", ".join(markers)
                        for r in p.runs[1:]:
                            r.text = ""


def repair_preference(src: Path, dst: Path, draft: Mapping[str, Any], plan: Mapping[str, Any], template: Path,
                      min_font: float) -> Dict[str, Any]:
    """Make the deck's allocation, section labels and order follow the Phase 2 plan (see module docstring)."""
    prs = Presentation(str(src))
    expected = _draft_slides(draft, plan)
    nums = _content_slide_numbers(prs)
    match = _match_slides(V.read_deck_bullets(src), nums, expected)
    targets = [match[n] for n in nums]
    one_to_one = (None not in targets and len(set(targets)) == len(targets) == len(expected))
    if not one_to_one:
        meta = G.generate_pptx(dict(draft), dict(plan), template, dst, min_font_size=min_font,
                               inputs={"plan": "(repair)", "draft": "(repair)"})
        return {"changed": True, "mode": "rebuild_from_plan_and_draft",
                "reason": (f"slide allocation does not match the plan (deck has {len(nums)} content slides, plan has "
                           f"{len(expected)}; unmatched={sum(t is None for t in targets)}, "
                           f"duplicates={len(targets) - len(set(targets))})"),
                "affected_slides": list(range(1, meta["n_slides"] + 1)), "retitled": [], "moved": [],
                "plan_allocation": dict(plan["slide_allocation"]), "plan_order": list(plan["section_order"])}
    retitled, moved = [], []
    for n, ei in zip(nums, targets):
        want = _planned_title(expected[ei])
        t = prs.slides[n - 1].shapes.title
        if t is not None and t.text_frame.text.strip() != want:
            retitled.append({"slide": n, "old": t.text_frame.text.strip(), "new": want})
            t.text_frame.text = want
    desired = sorted(range(len(nums)), key=lambda i: targets[i])          # deck positions in plan order
    if desired != list(range(len(nums))):
        lst = prs.slides._sldIdLst
        ids = list(lst)
        slots = [nums[i] - 1 for i in range(len(nums))]
        for i, pos in enumerate(slots):
            lst.remove(ids[pos])
        for pos, di in zip(slots, desired):
            lst.insert(pos, ids[nums[di] - 1])
        moved = [{"from_slide": nums[di], "to_slide": nums[i]} for i, di in enumerate(desired) if di != i]
    if not retitled and not moved:
        return {"changed": False, "mode": "in_place", "reason": "slide labels and order already follow the plan",
                "affected_slides": [], "retitled": [], "moved": []}
    _refresh_index_and_notes(prs, [(nums[i], expected[i]) for i in range(len(nums))])
    _save(prs, dst)
    return {"changed": True, "mode": "in_place", "retitled": retitled, "moved": moved,
            "affected_slides": sorted({r["slide"] for r in retitled} | {m["from_slide"] for m in moved} | {m["to_slide"] for m in moved}),
            "plan_allocation": dict(plan["slide_allocation"]), "plan_order": list(plan["section_order"])}


# =========================================================================================== #
# The loop                                                                                     #
# =========================================================================================== #
def _why(failures: Sequence[Mapping[str, Any]], family: str) -> List[str]:
    return [f"{f['metric']} = {f['value']} violates {f['criterion']}" for f in failures if f["family"] == family]


def run_round(cand: Mapping[str, Any], ver: Mapping[str, Any], active: Sequence[str], failures: Sequence[Mapping[str, Any]],
              ctx: Mapping[str, Any], work: Path, repairers: Mapping[str, Callable]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply every active failing family, in priority order, to a copy of ``cand``.  Returns (new candidate, actions)."""
    draft = copy.deepcopy(cand["draft"])
    cur = Path(cand["pptx"])
    actions: List[Dict[str, Any]] = []
    cfg_min_font = ctx["min_font_size"]

    def stage(name: str, family: str, fn: Callable, what_src: str, **kw):
        nonlocal cur
        try:
            res = fn(cur, work, **kw)
        except (ValueError, KeyError) as e:               # e.g. the draft cannot be rebuilt against the plan
            res = {"changed": False, "error": f"{type(e).__name__}: {e}"}
        act = {"repair_type": name, "family": family, "why": _why(failures, family) if family in FAMILIES else [],
               "source": what_src, "changed": bool(res["changed"]), "affected_slides": res.get("affected_slides", []),
               "affected_bullets": res.get("affected_bullets", []), "what": res}
        actions.append(act)
        if res["changed"]:
            cur = work
        return res

    if "overflow" in active:
        boxes = [b for b in ver["details"]["overflow"]["boxes"] if b["overflow"]]
        stage("overflow", "overflow", lambda s, d, **k: repairers["overflow"](s, d, boxes, cfg_min_font),
              "Phase 4 fitter (pptx_generator.fit_body) on the boxes flagged by the verifier")
    if "support" in active:
        stage("support_fidelity", "support",
              lambda s, d, **k: repairers["support"](s, d, ver, ctx["parsed"], draft, ctx["plan"], ctx["sr_threshold"]),
              "Phase 3 draft bullets / parsed source sentences at the bullets' source pointers")
    if "preference" in active:
        stage("preference", "preference",
              lambda s, d, **k: repairers["preference"](s, d, draft, ctx["plan"], ctx["template"], cfg_min_font),
              "Phase 2 plan (slide_allocation, section_order) + Phase 3 draft")
    if any(a["changed"] for a in actions if a["family"] in ("support", "preference")) and cur != Path(cand["pptx"]):
        # content / structure changed: re-fit whatever now overflows (same Phase 4 fitter; text untouched)
        ov = V.overflow_rate(cur, ctx["verify_kw"].get("min_font_size", V.DEFAULT_MIN_FONT_SIZE),
                             ctx["verify_kw"].get("fill_threshold", V.DEFAULT_FILL_THRESHOLD))
        boxes = [b for b in ov["boxes"] if b["overflow"]]
        if boxes:
            res = repairers["overflow"](cur, work, boxes, cfg_min_font)
            actions.append({"repair_type": "overflow", "family": "overflow",
                            "why": [f"{len(boxes)} box(es) overflow after content/structure was restored"],
                            "source": "Phase 4 fitter (pptx_generator.fit_body), re-fit after restoration",
                            "changed": bool(res["changed"]), "affected_slides": res.get("affected_slides", []),
                            "affected_bullets": [], "what": res})
            if res["changed"]:
                cur = work
    return {"draft": draft, "pptx": cur}, actions


def repair_loop(parsed: Mapping[str, Any], ref_qpp: Any, plan: Mapping[str, Any], draft: Mapping[str, Any],
                pptx_path: Path, out_dir: Path, *, template: Optional[Path] = None,
                config: Optional[RepairConfig] = None, verify_kw: Optional[Mapping[str, Any]] = None,
                repairers: Optional[Mapping[str, Callable]] = None, write: bool = True,
                inputs: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Bounded repair loop.  Writes final.pptx, final_verification.json, repair_history.json (+ candidates/) to
    ``out_dir`` and returns ``{"history", "verification", "pptx", "draft"}`` of the lowest-J candidate."""
    cfg = config or RepairConfig()
    vkw = dict(verify_kw or {})
    out_dir, pptx_path = Path(out_dir), Path(pptx_path)
    cdir = out_dir / "candidates"
    if write and cdir.exists():
        shutil.rmtree(cdir)
    reps = {"overflow": repair_overflow, "support": repair_support, "preference": repair_preference, **(repairers or {})}
    ctx = {"parsed": parsed, "plan": plan, "template": Path(template) if template else G_DEFAULT_TEMPLATE,
           "min_font_size": vkw.get("min_font_size", G.MIN_FONT_SIZE), "verify_kw": vkw,
           "sr_threshold": vkw.get("sr_threshold", V.DEFAULT_SR_THRESHOLD)}

    cur_draft = copy.deepcopy(dict(draft))

    start = {"draft": cur_draft, "pptx": pptx_path, "file": pptx_path.name}
    ver0 = V.verify(parsed, ref_qpp, plan, cur_draft, pptx_path, **vkw)
    j0 = ver0["objective"]
    cand, ver = start, ver0
    best, best_ver, best_round = start, ver0, 0
    rounds: List[Dict[str, Any]] = []
    blocked: List[str] = []
    stop = "max_repair_rounds_reached"
    for r in range(1, cfg.max_rounds + 1):
        failures = detect_failures(ver, cfg)
        if not failures:
            stop = "all_criteria_satisfied"
            break
        active = [f for f in FAMILIES if any(x["family"] == f for x in failures) and f not in blocked]
        if not active:
            stop = "no_unblocked_repair_type_available"
            break
        work = cdir / f"round_{r}.pptx"
        new, actions = run_round(cand, ver, active, failures, ctx, work, reps)
        tried = sorted({a["family"] for a in actions}, key=FAMILIES.index)
        rec: Dict[str, Any] = {"round": r, "detected_failures": [dict(f) for f in failures], "families_attempted": active,
                               "actions": actions, "previous_J": _r(ver["objective"]), "best_J_before": _r(best_ver["objective"])}
        if not any(a["changed"] for a in actions):
            rec.update(new_J=None, improved=False, became_best=False, candidate_file=None,
                       note="no repair changed the candidate (nothing repairable by the active families)")
            rounds.append(rec)
            blocked += [f for f in active if f not in blocked]
            stop = "no_effective_repair"
            break
        new["file"] = f"candidates/{work.name}"
        nver = V.verify(parsed, ref_qpp, plan, new["draft"], Path(new["pptx"]), **vkw)
        improved = nver["objective"] < ver["objective"] - 1e-12
        became_best = nver["objective"] < best_ver["objective"] - 1e-12
        rec.update(new_J=_r(nver["objective"]), improved=bool(improved), became_best=bool(became_best),
                   candidate_file=new["file"], metrics_after=_metrics(nver),
                   remaining_failures=[f["criterion"] for f in detect_failures(nver, cfg)])
        if became_best:
            best, best_ver, best_round = new, nver, r
        if improved:
            cand, ver = new, nver
        else:                                              # not an improvement: go back, block what was tried
            cand, ver = best, best_ver
            blocked += [f for f in active if f not in blocked]
            rec["note"] = "J did not decrease: candidate rejected; tried repair types blocked for later rounds"
        rec["blocked_after_round"] = list(blocked)
        rounds.append(rec)
    else:
        failures = detect_failures(ver, cfg)
        stop = "all_criteria_satisfied" if not failures else "max_repair_rounds_reached"

    history = {
        "schema": SCHEMA,
        "mode": "deterministic bounded repair loop (extractive; no LLM, no API)",
        "inputs": dict(inputs or {}),
        "config": {**asdict(cfg), "repair_priority": list(FAMILIES), "verify": vkw,
                   "objective": "J(S) = Dpref + mu*O + nu*(1-NF) + xi*(1-SR); lower is better",
                   "objective_weights": ver0["config"]["objective_weights"]},
        "initial": {"J": _r(j0), "metrics": _metrics(ver0), "failures": [f["criterion"] for f in detect_failures(ver0, cfg)]},
        "rounds": rounds, "rounds_used": len(rounds), "max_repair_rounds": cfg.max_rounds, "stop_reason": stop,
        "final": {"selected_round": best_round, "selected_candidate": best["file"] if best_round else "(initial deck, unchanged)",
                  "J": _r(best_ver["objective"]), "metrics": _metrics(best_ver),
                  "improved_over_initial": bool(best_ver["objective"] < j0 - 1e-12),
                  "J_change": _r(best_ver["objective"] - j0),
                  "repair_types_applied": sorted({a["repair_type"] for rr in rounds for a in rr["actions"] if a["changed"]
                                                  and rr["improved"] and rr["round"] <= best_round}),
                  "remaining_failures": [f["criterion"] for f in detect_failures(best_ver, cfg)],
                  "not_repairable_note": "Dstyle (QPP style distance) is not repaired: it would require changing content."},
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["pptx"], out_dir / "final.pptx")
        final_ver = dict(best_ver)
        final_ver["inputs"] = {**best_ver["inputs"], "pptx": str(out_dir / "final.pptx")}
        for name, obj in (("final_verification.json", final_ver), ("repair_history.json", history),
                          ("final_draft.json", best["draft"])):
            (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"history": history, "verification": best_ver, "pptx": Path(best["pptx"]), "draft": best["draft"]}


G_DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "data" / "reference_decks" / "synthetic_reference.pptx"

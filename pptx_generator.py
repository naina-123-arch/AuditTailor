"""
pptx_generator.py -- AuditTailor Phase 4: editable PPTX generation / template filling.

    Phase 2 plan.json  +  Phase 3 draft.json  +  template PPTX  ->  editable .pptx  +  pptx_metadata.json

Everything on the slides is real, editable PowerPoint text (title / body placeholders of the template's own
layouts).  The text comes ONLY from the Phase 3 draft: bullet text, section names, source pointers.  Nothing is
paraphrased, invented, padded or truncated.  No LLM, no API, no network, no randomness.

Template handling
    * slide size, slide master, theme and layouts of the template are kept untouched;
    * the template's example slides are removed (their content belongs to the reference paper) and new slides
      are created from the template's own "Title Slide" and "Title and Content" layouts, so title placement,
      body placement and the master's visual elements are inherited.

Source traceability (three redundant carriers)
    1. every bullet ends with a small grey inline marker  ``[S004, p.2]``  (same paragraph, same text frame);
    2. the speaker notes of every content slide list the full pointer of each bullet
       (sentence id, page, canonical section, original paper heading);
    3. an optional final "Source Index" slide lists the pointers per slide.

Overflow handling (basic, estimate-based)
    The fill ratio of each body box is estimated with the same text-fit estimator Phase 1 uses for eta
    (qpp.estimate_fill).  If it exceeds MAX_FILL: (1) the font size is reduced in FONT_STEP steps down to
    MIN_FONT_SIZE, then (2) paragraph spacing is tightened.  Text is never deleted.  Every adjustment, and
    every box that still does not fit at the minimum size, is recorded in the metadata for Phase 5.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.util import Pt

from qpp import TextMeasurer, estimate_fill, resolve_font_size

SCHEMA = "audittailor.pptx_metadata.v1"

# ------------------------------------------------------------------ configuration (module level, overridable) #
MIN_FONT_SIZE = 16          # pt: body text is never reduced below this
SOURCE_SLIDE_MIN_FONT_SIZE = 12   # pt: floor for the (dense, reference-only) Source Index slide
FONT_STEP = 2               # pt per reduction step
# estimate_fill is slightly optimistic vs PowerPoint render; keep headroom so text does not clip
MAX_FILL = 0.85             # estimated text height / box height above which a box counts as overflowing
TIGHT_SPACE_BEFORE_PT = 2   # paragraph spacing used when font reduction alone is not enough
TIGHT_LINE_SPACING = 0.95
MARKER_SCALE = 0.5          # source marker size relative to the bullet text
MARKER_MIN_PT = 10
MARKER_RGB = (0x59, 0x59, 0x59)
FIXED_TIMESTAMP = _dt.datetime(2000, 1, 1)   # keeps the .pptx byte-for-byte reproducible

_TITLE_TYPES = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE}
_BODY_TYPES = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT}
_SUBTITLE = PP_PLACEHOLDER.SUBTITLE


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def source_marker(src: Dict[str, Any]) -> str:
    """Compact pointer text, built only from Phase 3 fields: ``[S004, p.2]``."""
    return f"[{src['sentence_id']}, p.{src['page']}]"


def expected_content_slides(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The planner's slide sequence: section order x per-section allocation."""
    order = plan.get("section_order") or plan.get("presented_section_order") or []
    alloc = plan.get("slide_allocation") or {}
    out, n = [], 0
    for sec in order:
        k = int(alloc.get(sec, 0))
        for i in range(1, k + 1):
            n += 1
            out.append({"slide_number": n, "section": sec, "section_slide_index": i, "section_slide_count": k})
    return out


def _find_layout(prs, name: str, title_types, want_body: bool):
    for lay in prs.slide_layouts:
        if lay.name == name:
            return lay
    for lay in prs.slide_layouts:                      # fallback: first layout with the right placeholders
        types = {p.placeholder_format.type for p in lay.placeholders}
        if types & set(title_types) and (bool(types & _BODY_TYPES) == want_body):
            return lay
    raise ValueError(f"template has no usable layout for '{name}'")


def _placeholder(slide, types):
    for ph in slide.placeholders:
        if ph.placeholder_format.type in types:
            return ph
    return None


def _remove_all_slides(prs) -> int:
    """Drop the template's example slides (relationships included, so no orphan parts are saved)."""
    lst = prs.slides._sldIdLst
    ids = list(lst)
    for sld in ids:
        prs.part.drop_rel(sld.rId)
        lst.remove(sld)
    return len(ids)


def _set_marker_style(run, main_pt: float) -> None:
    run.font.size = Pt(max(MARKER_MIN_PT, round(main_pt * MARKER_SCALE)))
    run.font.color.rgb = RGBColor(*MARKER_RGB)
    run.font.bold = False


# --------------------------------------------------------------------------- #
# Body filling + overflow control
# --------------------------------------------------------------------------- #
def _fill_body(ph, items: List[Dict[str, Any]]) -> List[Any]:
    """items: [{"text", "marker"(optional), "level"}] -> one paragraph per item; returns (paragraph, marker_run).

    Clears any existing body content first so repeated calls (e.g. overflow split search)
    do not accumulate runs or paragraphs.
    """
    tf = ph.text_frame
    tf.clear()  # leaves a single empty paragraph
    pars = []
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = int(it.get("level", 0))
        r = p.add_run()
        r.text = it["text"]
        mr = None
        if it.get("marker"):
            mr = p.add_run()
            mr.text = "  " + it["marker"]
        pars.append((p, r, mr))
    return pars


def _apply_size(pars, base_sizes, target_pt: float) -> None:
    for (p, r, mr), base in zip(pars, base_sizes):
        size = min(base, target_pt)
        r.font.size = Pt(size)
        if mr is not None:
            _set_marker_style(mr, size)


def fit_body(ph, pars, measurer: TextMeasurer, min_font: float, slide_label: Any) -> Dict[str, Any]:
    """Estimate-based overflow control for one body placeholder.  Never removes or shortens text."""
    base = [float(resolve_font_size(ph, p, r)) for (p, r, _m) in pars]
    for (p, r, mr), b in zip(pars, base):             # marker style (size relative to the template size)
        if mr is not None:
            _set_marker_style(mr, b)
    start = max(base) if base else 0.0
    fill0 = estimate_fill(ph, measurer)
    rec: Dict[str, Any] = {
        "slide": slide_label, "shape": ph.name, "font_size_start_pt": start, "font_size_final_pt": start,
        "min_font_size_pt": min_font, "fill_before": None if fill0 is None else round(fill0, 4),
        "fill_after": None if fill0 is None else round(fill0, 4), "font_reduced": False,
        "spacing_tightened": False, "overflow_resolved": True, "adjusted": False,
    }
    if fill0 is None or fill0 <= MAX_FILL:
        return rec

    fill, size = fill0, start
    while fill > MAX_FILL and size - FONT_STEP >= min_font:
        size -= FONT_STEP
        _apply_size(pars, base, size)
        rec["font_reduced"] = True
        fill = estimate_fill(ph, measurer)
    if fill > MAX_FILL and size > min_font:          # last partial step down to exactly the floor
        size = float(min_font)
        _apply_size(pars, base, size)
        rec["font_reduced"] = True
        fill = estimate_fill(ph, measurer)
    if fill > MAX_FILL:                               # spacing as the second lever
        for (p, _r, _m) in pars:
            p.space_before = Pt(TIGHT_SPACE_BEFORE_PT)
            p.line_spacing = TIGHT_LINE_SPACING
        rec["spacing_tightened"] = True
        fill = estimate_fill(ph, measurer)

    rec["font_size_final_pt"] = min(start, size) if rec["font_reduced"] else start
    rec["fill_after"] = None if fill is None else round(fill, 4)
    rec["adjusted"] = rec["font_reduced"] or rec["spacing_tightened"]
    rec["overflow_resolved"] = fill is not None and fill <= MAX_FILL
    return rec


def _normalise_zip_timestamps(path: Path) -> None:
    """python-pptx stamps every zip entry with the current time; rewrite them with a fixed one so that two runs
    produce byte-identical files (entry order, names, contents and compression are unchanged)."""
    import os
    import zipfile
    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=(2000, 1, 1, 0, 0, 0))
            zi.compress_type = info.compress_type
            zi.external_attr = info.external_attr
            zout.writestr(zi, zin.read(info.filename))
    os.replace(tmp, path)


def _set_notes(slide, lines: List[str]) -> None:
    slide.notes_slide.notes_text_frame.text = "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def generate_pptx(draft: Dict[str, Any], plan: Dict[str, Any], template: Path, out_pptx: Path,
                  min_font_size: float = MIN_FONT_SIZE, source_slide: bool = True,
                  inputs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Build the deck, save it to ``out_pptx`` and return the metadata dict (also JSON-serialisable)."""
    template, out_pptx = Path(template), Path(out_pptx)
    if not template.is_file():
        raise FileNotFoundError(f"template PPTX not found: {template}")
    if min_font_size <= 0:
        raise ValueError("min_font_size must be > 0")

    errors: List[str] = []
    warnings: List[str] = []

    # ---- 1. reconcile plan and draft (never rebuild what Phase 2/3 decided) ----
    expected = expected_content_slides(plan)
    drafted = {(s["section"], s["section_slide_index"]): s for s in draft["slides"]}
    exp_keys = [(e["section"], e["section_slide_index"]) for e in expected]
    if len(draft["slides"]) != len(expected) or set(drafted) != set(exp_keys):
        raise ValueError(
            "Phase 3 draft does not match the Phase 2 plan: plan has "
            f"{len(expected)} content slides {[f'{s}#{i}' for s, i in exp_keys]}, draft has {len(draft['slides'])} "
            f"{[f'{s}#{i}' for s, i in sorted(drafted)]}")
    if [s["section"] for s in draft["slides"]] != [e["section"] for e in expected]:
        warnings.append("draft slide order differs from the plan's section order; the plan's order was used")
    warnings.extend(f"Phase 3: {w}" for w in draft.get("warnings", []))

    # ---- 2. load template, clear its example slides ----
    prs = Presentation(str(template))
    n_removed = _remove_all_slides(prs)
    lay_title = _find_layout(prs, "Title Slide", _TITLE_TYPES, want_body=False)
    lay_body = _find_layout(prs, "Title and Content", _TITLE_TYPES, want_body=True)
    measurer = TextMeasurer()
    if measurer.name.startswith("constant"):
        warnings.append("Pillow/fonts unavailable: overflow estimate uses a constant 0.5 em character width")

    slide_map: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    # ---- 3. title slide ----
    s = prs.slides.add_slide(lay_title)
    t_ph = _placeholder(s, _TITLE_TYPES)
    t_ph.text_frame.text = draft["title"]
    sub = _placeholder(s, {_SUBTITLE})
    parsed_src = Path((draft.get("inputs") or {}).get("parsed", "") or "").name
    if sub is not None:
        sub.text_frame.text = ("Source-grounded extractive draft" + (f" · {parsed_src}" if parsed_src else ""))
    else:
        warnings.append("title layout has no subtitle placeholder; subtitle omitted")
    slide_map.append({"pptx_slide": 1, "kind": "title", "section": None, "draft_slide_number": None,
                      "title": draft["title"], "n_bullets": 0, "n_source_pointers": 0})

    # ---- 4. content slides, in the planner's order ----
    pointer_rows: List[Dict[str, Any]] = []       # for the Source Index slide
    for e in expected:
        d = drafted[(e["section"], e["section_slide_index"])]
        bullets = d["bullets"]
        if not bullets:
            skipped.append({"draft_slide_number": d["slide_number"], "section": e["section"],
                            "reason": "Phase 3 selected no bullets; no empty slide is created"})
            warnings.append(f"draft slide {d['slide_number']} ({e['section']}) has no bullets -> skipped (no empty slide)")
            continue
        n = len(slide_map) + 1
        # Titles always use the canonical section name (+ index). Purpose is carried in
        # draft JSON and speaker notes so preference (Dpref) stays aligned with QPP.
        sec = e["section"]
        if e["section_slide_count"] > 1:
            title = f"{sec} ({e['section_slide_index']}/{e['section_slide_count']})"
        else:
            title = sec
        slide = prs.slides.add_slide(lay_body)
        _placeholder(slide, _TITLE_TYPES).text_frame.text = title
        body = _placeholder(slide, _BODY_TYPES)
        items = [{"text": b["text"], "marker": source_marker(b["source"]), "level": b.get("level", 0)} for b in bullets]
        visuals = d.get("visuals") or []
        pars = _fill_body(body, items)
        rec = fit_body(body, pars, measurer, min_font_size, n)
        overflow.append(rec)
        if not rec["overflow_resolved"]:
            warnings.append(f"slide {n}: text may still overflow at the minimum font size ({rec['fill_after']})")
        notes = [f"Section: {e['section']}", f"Purpose: {d.get('purpose') or e['section']}",
                 "Source pointers (from the Phase 3 draft):"]
        for i, b in enumerate(bullets, 1):
            sp = b["source"]
            notes.append(f"{i}. {sp['sentence_id']} — page {sp['page']}, section {sp['section']}, "
                         f"heading \"{sp['heading']}\"")
        for v in visuals:
            notes.append(f"Visual: {v.get('text')} (page {v.get('page')})")
        _set_notes(slide, notes)
        slide_map.append({
            "pptx_slide": n, "kind": "content", "section": e["section"], "draft_slide_number": d["slide_number"],
            "title": title, "purpose": d.get("purpose"), "n_bullets": len(bullets),
            "n_source_pointers": len(bullets),
            "bullets": [{"text": b["text"], "source": dict(b["source"]), "marker": source_marker(b["source"])}
                        for b in bullets],
            "visuals": list(visuals),
        })
        pointer_rows.append({"slide": n, "section": e["section"], "markers": [source_marker(b["source"]) for b in bullets]})

    # ---- 5. optional Source Index slide (pointers only; derived from the draft, nothing new) ----
    if source_slide and pointer_rows:
        n = len(slide_map) + 1
        slide = prs.slides.add_slide(lay_body)
        _placeholder(slide, _TITLE_TYPES).text_frame.text = "Source Index"
        body = _placeholder(slide, _BODY_TYPES)
        items = [{"text": f"Slide {r['slide']} · {r['section']}: " + ", ".join(r["markers"])} for r in pointer_rows]
        pars = _fill_body(body, items)
        rec = fit_body(body, pars, measurer, min(min_font_size, SOURCE_SLIDE_MIN_FONT_SIZE), n)
        overflow.append(rec)
        if not rec["overflow_resolved"]:
            warnings.append(f"slide {n} (Source Index): text may still overflow at the minimum font size")
        _set_notes(slide, ["[Sxxx, p.N] = Phase 3 sentence id and page in the source paper."])
        slide_map.append({"pptx_slide": n, "kind": "source_index", "section": None, "draft_slide_number": None,
                          "title": "Source Index", "n_bullets": 0, "n_source_pointers": 0})

    # ---- 6. save (fixed timestamps -> reproducible bytes) ----
    cp = prs.core_properties
    cp.title, cp.author, cp.last_modified_by = draft["title"], "AuditTailor", "AuditTailor"
    cp.created = cp.modified = cp.last_printed = FIXED_TIMESTAMP
    cp.revision = 1
    out_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_pptx))
    _normalise_zip_timestamps(out_pptx)

    content = [m for m in slide_map if m["kind"] == "content"]
    n_bullets = sum(m["n_bullets"] for m in content)
    n_ptr = sum(m["n_source_pointers"] for m in content)
    planned_bullets = sum(len(x["bullets"]) for x in draft["slides"])
    if n_bullets != planned_bullets:   # skipped slides carry no bullets, so the totals must agree
        errors.append(f"bullet count mismatch: draft {planned_bullets}, slides {n_bullets}")
    adjusted = [o for o in overflow if o["adjusted"]]
    return {
        "schema": SCHEMA,
        "mode": "template filling (python-pptx), extractive text only: no LLM, no API",
        "inputs": {"template": str(template), **(inputs or {})},
        "output_pptx": str(out_pptx),
        "template_info": {
            "slide_width_emu": int(prs.slide_width), "slide_height_emu": int(prs.slide_height),
            "example_slides_removed": n_removed, "title_layout": lay_title.name, "content_layout": lay_body.name,
        },
        "config": {"min_font_size_pt": min_font_size, "max_fill": MAX_FILL, "font_step_pt": FONT_STEP,
                   "source_slide": bool(source_slide), "text_measurer": measurer.name},
        "n_slides": len(prs.slides),
        "n_content_slides": len(content),
        "n_content_slides_planned": len(expected),
        "section_order": [e["section"] for e in expected if e["section_slide_index"] == 1],
        "slide_to_section": [{"pptx_slide": m["pptx_slide"], "kind": m["kind"], "section": m["section"],
                              "title": m["title"], "draft_slide_number": m["draft_slide_number"]} for m in slide_map],
        "bullets_per_slide": {str(m["pptx_slide"]): m["n_bullets"] for m in content},
        "n_bullets": n_bullets,
        "n_bullets_in_draft": planned_bullets,
        "n_source_pointers": n_ptr,
        "all_bullets_have_source_pointer": n_bullets == n_ptr,
        "slides": slide_map,
        "skipped_empty_slides": skipped,
        "overflow_checks": overflow,
        "overflow_adjustments": adjusted,
        "n_overflow_adjustments": len(adjusted),
        "n_unresolved_overflow": sum(1 for o in overflow if not o["overflow_resolved"]),
        "warnings": warnings,
        "errors": errors,
    }

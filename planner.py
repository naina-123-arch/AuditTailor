"""
planner.py -- AuditTailor Phase 2, Stage 3 (Budgeted Planning), proposal Sec. 4.3

Turns the presenter's emphasis (QPP from Phase 1) and the target paper's own
structure into an INTEGER number of slides per canonical section, plus the
section order.  Pure Python (standard library only): no parsing, no TF-IDF/MMR,
no slide generation, no verification/repair -- those are later phases.

Notation (proposal Sec. 4.3 / Appendix A)
-----------------------------------------
    l_k   words in section k of the target paper (Phase 1: label_word_counts_l)
    P     sections that actually occur in the paper  {k : l_k > 0}
    p_k   natural share            p_k  = l_k / sum_{j in P} l_j
    w_k   presenter emphasis       (Phase 1 QPP, theta.w)
    w'_k  restricted + renormalised w'_k = w_k / sum_{j in P} w_j      (k in P)
          (if that sum is 0, w'_k = p_k -- the proposal's fallback)
    q_k   blend                    q_k  = lambda * w'_k + (1 - lambda) * p_k   Eq. (3)
    n_k   integer slides           largest-remainder rounding of N * q_k
    pi    presenter section order  (Phase 1 QPP, theta.pi)

Algorithm
---------
 1. Validate N (integer >= 1) and lambda (finite number in [0, 1]).
 2. P = {k : l_k > 0}; p_k = l_k / sum_P l_j.
 3. w'_k = w_k / sum_P w_j  (sections absent from the paper are dropped and the
    rest renormalised; if sum_P w_j = 0 fall back to w'_k = p_k).
 4. q_k = lambda*w'_k + (1-lambda)*p_k.   lambda = 0 -> paper only; lambda = 1 -> presenter only.
 5. Largest remainder (Hamilton) rounding of N*q_k:
        raw_k  = N * q_k
        base_k = floor(raw_k)
        m      = N - sum_k base_k           (0 <= m < |P|, since the remainders sum to m)
        the m sections with the largest remainder raw_k - base_k receive +1 slide;
        ties are broken by the larger q_k, then by canonical section order
        (so the result is fully deterministic).
    Consequences (proved in the proposal, tested here): sum_k n_k = N exactly and
    |n_k - N*q_k| < 1.  A section with n_k = 0 is simply not presented (it will
    lower coverage later); no minimum is forced, because that would depart from
    the proposal.  Sections absent from the paper always get 0 slides.
 6. Order: follow pi, skipping sections not in P; a section in P that is missing
    from pi is inserted before Conclusion (appended at the end if Conclusion is
    not in the order).

All arithmetic is done with exact rationals (fractions.Fraction) so that ties in
step 5 are decided by the numbers themselves and never by floating-point noise
(e.g. 10 * 0.18 == 1.7999999999999998 in floats).  Floats are only produced when
the plan is serialised.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence

from section_tagger import (
    CANONICAL_SECTIONS,
    detect_purpose,
    purpose_title,
    PURPOSE_TITLES,
)

SECTIONS: List[str] = list(CANONICAL_SECTIONS)
_ORDER_INDEX = {k: i for i, k in enumerate(SECTIONS)}


# --------------------------------------------------------------------------- #
# Validation / conversion helpers
# --------------------------------------------------------------------------- #
def _to_fraction(x: Any, name: str) -> Fraction:
    """Exact rational from an int / float / Fraction (floats via their shortest decimal repr)."""
    if isinstance(x, bool) or not isinstance(x, (numbers.Real, Fraction)):
        raise ValueError(f"{name} must be a real number, got {x!r}")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, numbers.Integral):
        return Fraction(int(x))
    xf = float(x)
    if not math.isfinite(xf):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return Fraction(repr(xf))


def validate_lambda(lam: Any) -> Fraction:
    """lambda must be a finite real number with 0 <= lambda <= 1."""
    f = _to_fraction(lam, "lambda")
    if not (0 <= f <= 1):
        raise ValueError(f"lambda must be in [0, 1], got {lam!r}")
    return f


def validate_n_slides(n: Any) -> int:
    """N must be an integer >= 1 (bool and non-integral floats are rejected)."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise ValueError(f"total number of slides N must be an integer, got {n!r}")
    if int(n) < 1:
        raise ValueError(f"total number of slides N must be >= 1, got {n!r}")
    return int(n)


def _clean_word_counts(section_words: Mapping[str, Any]) -> Dict[str, int]:
    out = {k: 0 for k in SECTIONS}
    for k, v in section_words.items():
        if k not in out:
            raise ValueError(f"unknown section label {k!r}; expected one of {SECTIONS}")
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"word count for {k} must be a number, got {v!r}")
        if not math.isfinite(float(v)) or float(v) < 0 or float(v) != int(v):
            raise ValueError(f"word count for {k} must be a non-negative integer, got {v!r}")
        out[k] = int(v)
    return out


def _clean_weights(w: Mapping[str, Any]) -> Dict[str, Fraction]:
    out = {k: Fraction(0) for k in SECTIONS}
    for k, v in w.items():
        if k not in out:
            raise ValueError(f"unknown section label {k!r} in reference weights; expected one of {SECTIONS}")
        f = _to_fraction(v, f"w[{k}]")
        if f < 0:
            raise ValueError(f"reference weight w[{k}] must be >= 0, got {v!r}")
        out[k] = f
    return out


def _clean_order(pi: Sequence[str]) -> List[str]:
    out: List[str] = []
    for k in pi:
        if k not in _ORDER_INDEX:
            raise ValueError(f"unknown section label {k!r} in order pi; expected one of {SECTIONS}")
        if k not in out:                       # keep first occurrence only
            out.append(k)
    return out


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def paper_distribution(words: Mapping[str, int]) -> Dict[str, Fraction]:
    """p_k = l_k / sum_{j in P} l_j  (0 for sections that do not occur)."""
    total = sum(words.values())
    if total <= 0:
        raise ValueError("the target paper has no words in any canonical section; nothing to plan")
    return {k: Fraction(words[k], total) for k in SECTIONS}


def restrict_and_renormalise(w: Mapping[str, Fraction], p: Mapping[str, Fraction],
                             present: Sequence[str]) -> "tuple[Dict[str, Fraction], bool]":
    """
    w'_k = w_k / sum_{j in P} w_j for k in P (0 elsewhere).
    Returns (w_prime, fell_back).  If sum_{j in P} w_j == 0 the proposal says w'_k = p_k.
    """
    s = sum((w[k] for k in present), Fraction(0))
    if s == 0:
        return {k: (p[k] if k in present else Fraction(0)) for k in SECTIONS}, True
    return {k: (w[k] / s if k in present else Fraction(0)) for k in SECTIONS}, False


def blend(w_prime: Mapping[str, Fraction], p: Mapping[str, Fraction], lam: Fraction,
          present: Sequence[str]) -> Dict[str, Fraction]:
    """Eq. (3):  q_k = lambda * w'_k + (1 - lambda) * p_k   for k in P (0 elsewhere)."""
    return {k: (lam * w_prime[k] + (1 - lam) * p[k] if k in present else Fraction(0)) for k in SECTIONS}


def largest_remainder(q: Mapping[str, Fraction], n_total: int, present: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Largest-remainder (Hamilton) rounding of N * q_k over the sections in `present`.

    Returns {section: {"raw": Fraction, "floor": int, "remainder": Fraction,
                       "extra": bool, "slides": int}} for every canonical section
    (sections not in `present` get raw = 0, slides = 0).
    """
    if sum((q[k] for k in present), Fraction(0)) != 1:
        raise ValueError("blended shares must sum to exactly 1 over the present sections")
    detail: Dict[str, Dict[str, Any]] = {}
    for k in SECTIONS:
        if k in present:
            raw = n_total * q[k]
            fl = raw.numerator // raw.denominator
            detail[k] = {"raw": raw, "floor": fl, "remainder": raw - fl, "extra": False}
        else:
            detail[k] = {"raw": Fraction(0), "floor": 0, "remainder": Fraction(0), "extra": False}

    m = n_total - sum(d["floor"] for d in detail.values())
    # largest remainder first; ties -> larger q_k; remaining ties -> canonical order
    ranked = sorted(present, key=lambda k: (-detail[k]["remainder"], -q[k], _ORDER_INDEX[k]))
    for k in ranked[:m]:
        detail[k]["extra"] = True
    for k in SECTIONS:
        detail[k]["slides"] = detail[k]["floor"] + (1 if detail[k]["extra"] else 0)
    assert sum(d["slides"] for d in detail.values()) == n_total   # guaranteed by construction
    return detail


def order_sections(pi: Sequence[str], present: Sequence[str]) -> "tuple[List[str], List[str]]":
    """
    Section order: follow pi, skip sections not in P; sections in P that are
    missing from pi are inserted before Conclusion (appended if Conclusion is not
    in the order).  Returns (order, inserted_sections).
    """
    pset = set(present)
    order = [k for k in pi if k in pset]
    missing = [k for k in SECTIONS if k in pset and k not in order]      # canonical order
    middle = [k for k in missing if k != "Conclusion"]
    if "Conclusion" in order:
        i = order.index("Conclusion")
        order[i:i] = middle
    else:
        order.extend(middle)
        if "Conclusion" in missing:
            order.append("Conclusion")
    return order, missing


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
def _f(x: Fraction, nd: int = 6) -> float:
    return round(float(x), nd)



def build_story_plan(
    parsed: Mapping[str, Any],
    allocation: Mapping[str, int],
    section_order: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Turn QPP slide allocation + the paper's real headings into a purpose-driven
    slide story.  Does not invent sections the paper lacks.
    """
    by_label: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SECTIONS}
    for sec in parsed.get("sections") or []:
        if sec.get("excluded"):
            continue
        lab = sec.get("label")
        if lab not in by_label:
            continue
        n_words = int(sec.get("n_words") or 0)
        if n_words <= 0 and not (sec.get("heading") or "").strip():
            continue
        by_label[lab].append({
            "heading": (sec.get("heading") or "").strip(),
            "n_words": n_words,
            "page": sec.get("page"),
            "index": sec.get("index"),
        })

    story: List[Dict[str, Any]] = []
    for label in section_order:
        n = int(allocation.get(label, 0))
        if n <= 0:
            continue
        paper_secs = by_label.get(label) or []
        purposes: List[Dict[str, Any]] = []
        seen_p: set = set()
        for ps in paper_secs:
            pur = detect_purpose(ps["heading"], label)
            if pur in seen_p:
                for p in purposes:
                    if p["purpose"] == pur:
                        if ps["heading"] and ps["heading"] not in p["source_headings"]:
                            p["source_headings"].append(ps["heading"])
                        p["n_words"] += ps["n_words"]
                        break
                continue
            seen_p.add(pur)
            purposes.append({
                "purpose": pur,
                "title": purpose_title(pur, ps["heading"]),
                "source_headings": [ps["heading"]] if ps["heading"] else [],
                "n_words": ps["n_words"],
                "canonical_section": label,
            })
        if not purposes:
            pur = detect_purpose("", label)
            purposes.append({
                "purpose": pur,
                "title": PURPOSE_TITLES.get(pur, label),
                "source_headings": [],
                "n_words": 0,
                "canonical_section": label,
            })

        if len(purposes) >= n:
            ranked = sorted(range(len(purposes)), key=lambda i: (-purposes[i]["n_words"], i))
            chosen_idx = sorted(ranked[:n])
            chosen = [purposes[i] for i in chosen_idx]
        else:
            chosen = list(purposes)
            extra = n - len(purposes)
            ranked = sorted(range(len(purposes)), key=lambda i: (-purposes[i]["n_words"], i))
            for j in range(extra):
                base = purposes[ranked[j % len(ranked)]]
                chosen.append({
                    "purpose": base["purpose"],
                    "title": base["title"],
                    "source_headings": list(base["source_headings"]),
                    "n_words": base["n_words"],
                    "canonical_section": label,
                })

        purpose_counts: Dict[str, int] = {}
        purpose_seen: Dict[str, int] = {}
        for c in chosen:
            purpose_counts[c["purpose"]] = purpose_counts.get(c["purpose"], 0) + 1
        for c in chosen:
            p = c["purpose"]
            purpose_seen[p] = purpose_seen.get(p, 0) + 1
            idx = purpose_seen[p]
            total = purpose_counts[p]
            title = c["title"]
            if total > 1:
                title = f"{title} ({idx}/{total})"
            prefers_evidence = p in (
                "main_results", "comparison", "ablation", "analysis", "metrics"
            )
            story.append({
                "purpose": p,
                "title": title,
                "canonical_section": label,
                "source_headings": c["source_headings"],
                "prefers_evidence": prefers_evidence,
                "section_slide_index": len([s for s in story if s["canonical_section"] == label]) + 1,
            })

    counts: Dict[str, int] = {}
    for s in story:
        counts[s["canonical_section"]] = counts.get(s["canonical_section"], 0) + 1
    for s in story:
        s["section_slide_count"] = counts[s["canonical_section"]]
    return story


@dataclass
class Plan:
    total_slides: int
    lam: Fraction
    word_counts: Dict[str, int]
    present: List[str]
    p: Dict[str, Fraction]
    w_reference: Dict[str, Fraction]
    w_prime: Dict[str, Fraction]
    q: Dict[str, Fraction]
    detail: Dict[str, Dict[str, Any]]
    order: List[str]
    inserted: List[str]
    fallbacks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    story: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def allocation(self) -> Dict[str, int]:
        return {k: self.detail[k]["slides"] for k in SECTIONS}

    @property
    def absent(self) -> List[str]:
        return [k for k in SECTIONS if k not in self.present]

    @property
    def presented_order(self) -> List[str]:
        return [k for k in self.order if self.detail[k]["slides"] > 0]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema": "audittailor.plan.v1",
            "total_slides": self.total_slides,
            "lambda": _f(self.lam, 12),
            "paper_word_counts": dict(self.word_counts),
            "present_sections": list(self.present),
            "absent_sections": self.absent,
            "paper_distribution": {k: _f(self.p[k]) for k in SECTIONS},
            "reference_distribution": {k: _f(self.w_reference[k]) for k in SECTIONS},
            "reference_distribution_restricted": {k: _f(self.w_prime[k]) for k in SECTIONS},
            "blended_distribution": {k: _f(self.q[k]) for k in SECTIONS},
            "slide_allocation": self.allocation,
            "section_order": list(self.order),
            "presented_section_order": self.presented_order,
            "inserted_sections": list(self.inserted),
            "allocation_detail": {
                k: {
                    "raw_N_times_q": _f(d["raw"]),
                    "floor": d["floor"],
                    "remainder": _f(d["remainder"]),
                    "extra_slide": d["extra"],
                    "slides": d["slides"],
                }
                for k, d in self.detail.items()
            },
            "fallbacks": list(self.fallbacks),
            "warnings": list(self.warnings),
        }
        if self.story:
            d["story"] = list(self.story)
        return d



# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def plan_slides(section_words: Mapping[str, Any], w: Mapping[str, Any], pi: Sequence[str],
                n_slides: Any, lam: Any) -> Plan:
    """
    Budgeted plan for a target paper.

    section_words : {label: l_k}  words per canonical label of the target paper
    w             : {label: w_k}  QPP emphasis distribution (Phase 1)
    pi            : QPP section order (Phase 1)
    n_slides      : N, desired number of content slides (integer >= 1)
    lam           : lambda in [0, 1]
    """
    lam_f = validate_lambda(lam)
    n = validate_n_slides(n_slides)
    words = _clean_word_counts(section_words)
    w_ref = _clean_weights(w)
    order_in = _clean_order(pi)

    p = paper_distribution(words)
    present = [k for k in SECTIONS if words[k] > 0]            # P, canonical order
    w_prime, fell_back = restrict_and_renormalise(w_ref, p, present)
    q = blend(w_prime, p, lam_f, present)
    detail = largest_remainder(q, n, present)
    order, inserted = order_sections(order_in, present)

    fallbacks: List[str] = []
    warnings: List[str] = []
    if fell_back:
        fallbacks.append("w' fell back to the paper's natural shares p_k because the reference has zero weight "
                         "on every section present in the paper (proposal Sec. 4.3)")
    for k in SECTIONS:
        if k not in present and w_ref[k] > 0:
            warnings.append(f"reference emphasises {k} (w={_f(w_ref[k])}) but the paper has no {k} section; "
                            f"its weight was dropped and the rest renormalised")
    for k in present:
        if detail[k]["slides"] == 0:
            warnings.append(f"{k} received 0 slides (N={n} is small for {len(present)} sections); "
                            f"it will not be presented")
    if n < len(present):
        warnings.append(f"N={n} is smaller than the number of present sections ({len(present)})")
    if not order_in:
        warnings.append("reference order pi is empty; sections were ordered canonically with Conclusion last")
    if inserted and order_in:
        warnings.append(f"sections {inserted} are in the paper but not in pi; inserted before Conclusion")

    return Plan(n, lam_f, words, present, p, w_ref, w_prime, q, detail, order, inserted, fallbacks, warnings)


def plan_from_phase1(parsed: Mapping[str, Any], qpp: Mapping[str, Any], n_slides: Any, lam: Any) -> Plan:
    """Plan from the two Phase 1 JSON documents (parsed paper + QPP).

    Also builds a purpose-driven story plan from the paper's real headings so
    that later phases can produce coherent scientific slides rather than
    generic section dumps.  QPP still controls the integer slide budget.
    """
    try:
        words = parsed["label_word_counts_l"]
        w = qpp["theta"]["w"]
        pi = qpp["theta"]["pi"]
    except KeyError as e:
        raise ValueError(f"Phase 1 JSON is missing the expected field {e}") from None
    plan = plan_slides(words, w, pi, n_slides, lam)
    try:
        story = build_story_plan(parsed, plan.allocation, plan.presented_order)
        plan.story = story
    except Exception as e:
        plan.warnings.append(f"story plan skipped: {e}")
    return plan

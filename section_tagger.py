"""
section_tagger.py -- AuditTailor Phase 1

Rule-based mapping from section headings / slide titles to the K = 6 canonical
section labels of the proposal (Sec. 4.2):

    Motivation, Background, Method, Setup, Results, Conclusion

Tagging is keyword based and deterministic (no LLM).  The ORDER of the rules is
the "priority order" the proposal asks for: the word "experimental" in
"Experimental Results" must not turn the heading into Setup, so Results rules
are tried before the generic Setup rule.

Titles that match no label return None.  Callers apply the proposal's
inheritance rule ("a slide whose title matches nothing inherits the label of the
previous slide"); see `inherit_labels`.

Titles that are not content at all (outline, references, thank-you, ...) are
detected separately by `excluded_kind`, because the proposal excludes title,
outline and reference slides.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

CANONICAL_SECTIONS: List[str] = [
    "Motivation",
    "Background",
    "Method",
    "Setup",
    "Results",
    "Conclusion",
]

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
_NUM_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivx]+\.)\s+")


def normalise_title(title: str) -> str:
    """Lower-case, drop a leading section number, '&' -> 'and', strip punctuation."""
    t = (title or "").lower().replace("&", " and ")
    t = _NUM_PREFIX.sub("", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- #
# Canonical-label rules.  FIRST MATCH WINS -> the order below is the priority.
# --------------------------------------------------------------------------- #
_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    (label, re.compile(pattern))
    for label, pattern in [
        # 1. Conclusion
        ("Conclusion",
         r"\b(conclusions?|concluding|future work|summary|takeaways?|take home|closing remarks)\b"),
        # 2. Setup -- only the *specific* phrases; must precede Results so that
        #    "Evaluation Metrics" / "Dataset and ..." are Setup.
        ("Setup",
         r"\b(experimental|experiment|evaluation)\s+(setups?|settings?|details|protocols?|design|configuration)\b"
         r"|\bimplementation details?\b"
         r"|\b(datasets?|data sets?|corpus|corpora|baselines?|metrics?|training details?|hyper ?parameters?)\b"),
        # 3. Results -- must precede the generic Setup rule ("Experimental Results")
        ("Results",
         r"\b(results?|analysis|analyses|evaluation|findings|performance|ablations?|discussion|"
         r"comparisons?|case stud(?:y|ies)|error analysis|benchmark(?:s|ing)?)\b"),
        # 4. Setup -- generic words
        ("Setup",
         r"\b(experiments?|experimental|setup|set up|settings?)\b"),
        # 5. Method
        ("Method",
         r"\b(methods?|methodology|approach(?:es)?|proposed|models?|architecture|framework|"
         r"algorithms?|formulation|pipeline|system|design|technique)\b"),
        # 6. Background
        ("Background",
         r"\b(background|related works?|prior work|previous work|preliminar(?:y|ies)|literature|state of the art)\b"),
        # 7. Motivation
        ("Motivation",
         r"\b(introduction|motivation|overview|problem|challenges?|goals?|objectives?|contributions?|why)\b"),
    ]
]

# --------------------------------------------------------------------------- #
# Non-content titles (slides / paper sections that are never labelled)
# --------------------------------------------------------------------------- #
_EXCLUDED: List[Tuple[str, "re.Pattern[str]"]] = [
    (kind, re.compile(pattern))
    for kind, pattern in [
        ("abstract", r"^abstract$"),
        ("outline",
         r"^(talk |presentation )?(outline|agenda|roadmap|contents|table of contents|plan)"
         r"( of (the |this )?(talk|presentation))?$"),
        ("references",
         r"^(references?|bibliography|further reading|selected references|works cited)$"),
        ("closing",
         r"^(thank you|thanks|questions|q and a|q a|acknowledg(?:e)?ments?|the end)\b"),
        ("appendix", r"^(appendix|appendices|backup|supplementary)\b"),
    ]
]


def tag_title(title: str) -> Optional[str]:
    """Return the canonical label for a heading/slide title, or None if no rule matches."""
    norm = normalise_title(title)
    if not norm:
        return None
    for label, pat in _RULES:
        if pat.search(norm):
            return label
    return None


def excluded_kind(title: str) -> Optional[str]:
    """Return 'abstract' | 'outline' | 'references' | 'closing' | 'appendix', or None."""
    norm = normalise_title(title)
    if not norm:
        return None
    for kind, pat in _EXCLUDED:
        if pat.search(norm):
            return kind
    return None


def inherit_labels(titles: Sequence[str], default: str = "Motivation") -> List[Tuple[str, str]]:
    """
    Label a sequence of titles in order, applying the proposal's inheritance rule.

    Returns a list of (label, source) where source is one of
      'title'     -- a keyword rule matched the title
      'inherited' -- no match; label copied from the previous element
      'default'   -- no match and nothing before it; `default` is used
    """
    out: List[Tuple[str, str]] = []
    prev: Optional[str] = None
    for t in titles:
        lab = tag_title(t)
        if lab is not None:
            out.append((lab, "title"))
            prev = lab
        elif prev is not None:
            out.append((prev, "inherited"))
        else:
            out.append((default, "default"))
            prev = default
    return out



# --------------------------------------------------------------------------- #
# Slide-purpose detection (story planning; finer than the 6 canonical labels)
# --------------------------------------------------------------------------- #
# Purposes are optional refinements used only when the paper heading provides
# enough evidence.  They never invent content and never replace the canonical
# labels used by QPP / planner budget allocation.

SLIDE_PURPOSES: List[str] = [
    "problem",
    "motivation",
    "research_gap",
    "objective",
    "related_work",
    "approach",
    "architecture",
    "methodology",
    "dataset",
    "experimental_setup",
    "metrics",
    "main_results",
    "comparison",
    "ablation",
    "analysis",
    "limitations",
    "future_work",
    "conclusion",
]

PURPOSE_TITLES: Dict[str, str] = {
    "problem": "Problem",
    "motivation": "Motivation",
    "research_gap": "Research Gap",
    "objective": "Objective",
    "related_work": "Related Work",
    "approach": "Proposed Approach",
    "architecture": "Architecture",
    "methodology": "Methodology",
    "dataset": "Dataset",
    "experimental_setup": "Experimental Setup",
    "metrics": "Evaluation Metrics",
    "main_results": "Main Results",
    "comparison": "Comparison",
    "ablation": "Ablation Study",
    "analysis": "Analysis",
    "limitations": "Limitations",
    "future_work": "Future Work",
    "conclusion": "Conclusion",
}

_PURPOSE_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    (purpose, re.compile(pattern))
    for purpose, pattern in [
        ("ablation", r"\bablation\b"),
        ("comparison", r"\b(comparison|comparisons|baseline|baselines|vs\.?|versus)\b"),
        ("analysis", r"\b(analysis|analyses|discussion|error analysis|case stud)\b"),
        ("main_results", r"\b(results?|findings|performance|evaluation)\b"),
        ("limitations", r"\blimitations?\b"),
        ("future_work", r"\bfuture work\b"),
        ("conclusion", r"\b(conclusions?|concluding|summary|takeaways?)\b"),
        ("architecture", r"\b(architecture|pipeline|system overview|framework overview)\b"),
        ("methodology", r"\b(methodology|methods?|algorithm|formulation)\b"),
        ("approach", r"\b(approach|proposed|model)\b"),
        ("dataset", r"\b(datasets?|data sets?|corpus|corpora)\b"),
        ("metrics", r"\b(metrics?|evaluation metrics?)\b"),
        ("experimental_setup", r"\b(experimental|experiment|setup|implementation|hyper ?parameters?)\b"),
        ("related_work", r"\b(related works?|prior work|previous work|literature|state of the art)\b"),
        ("research_gap", r"\b(gap|challenge|open problem|limitations of prior)\b"),
        ("objective", r"\b(objective|goal|research question|contributions?)\b"),
        ("problem", r"\b(problem|problem statement|problem definition)\b"),
        ("motivation", r"\b(introduction|motivation|overview|why)\b"),
    ]
]

_DEFAULT_PURPOSE_FOR_LABEL: Dict[str, str] = {
    "Motivation": "motivation",
    "Background": "related_work",
    "Method": "methodology",
    "Setup": "experimental_setup",
    "Results": "main_results",
    "Conclusion": "conclusion",
}


def detect_purpose(heading: str, canonical_label: Optional[str] = None) -> str:
    """Map a paper heading (and optional canonical label) to a slide purpose."""
    norm = normalise_title(heading or "")
    if norm:
        for purpose, pat in _PURPOSE_RULES:
            if pat.search(norm):
                return purpose
    if canonical_label and canonical_label in _DEFAULT_PURPOSE_FOR_LABEL:
        return _DEFAULT_PURPOSE_FOR_LABEL[canonical_label]
    return "motivation"


def purpose_title(purpose: str, heading: Optional[str] = None) -> str:
    """Human-readable slide title for a purpose."""
    base = PURPOSE_TITLES.get(purpose, purpose.replace("_", " ").title())
    if heading:
        h = (heading or "").strip()
        if 3 <= len(h) <= 60 and not re.match(r"^\d+(\.\d+)*\.?\s*$", h):
            h_clean = _NUM_PREFIX.sub("", h).strip()
            if h_clean and len(h_clean) <= 55:
                return h_clean
    return base


if __name__ == "__main__":  # tiny self-check
    cases = {
        "1 Introduction": "Motivation",
        "Related Work": "Background",
        "3.1 Proposed Method": "Method",
        "Experimental Setup": "Setup",
        "Experimental Results": "Results",   # the collision named in the proposal
        "Datasets and Baselines": "Setup",
        "Evaluation Metrics": "Setup",
        "Ablation Study": "Results",
        "Conclusion and Future Work": "Conclusion",
        "Model Architecture": "Method",
        "Some Random Title": None,
    }
    bad = {k: (tag_title(k), v) for k, v in cases.items() if tag_title(k) != v}
    print("OK" if not bad else f"MISMATCH: {bad}")
    assert detect_purpose("Model Architecture", "Method") == "architecture"
    assert detect_purpose("Ablation Study", "Results") == "ablation"
    assert detect_purpose("Experimental Results", "Results") == "main_results"
    print("purpose OK")

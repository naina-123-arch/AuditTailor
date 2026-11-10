"""
extractor.py -- AuditTailor Phase 3: key-free extractive drafting (TF-IDF + MMR).

Implements the extractive mode of Stage 4 (proposal Sec. 4.4, Eq. (4), Appendix B and C).
NO language model, NO API, NO key, NO training: every bullet is a sentence that already
exists in the target paper (optionally trimmed by a few conservative string rules) and
carries a source pointer (sentence_id, page, section).

Pipeline for one paper
----------------------
 1. Candidates     parsed sentences with a canonical label that pass simple "is this a real
                   body sentence?" filters (no captions / title / abstract / references /
                   fragments).                                              -> load_candidates
 2. TF-IDF         one vectoriser fitted on the candidate sentences (IDF over the sentences
                   of the paper, each sentence = one document).            -> build_tfidf
 3. Relevance      Rel(s) in [0,1] = weighted average of four signals       -> relevance_scores
                     topical   cosine(sentence, title + abstract)
                     keyword   mean TF-IDF weight of the sentence's terms / max over the paper
                     section   q_k / max_j q_j  (blended share from the Phase 2 plan)
                     evidence  1 if the sentence has a number together with a result word
 4. MMR            per section, one sentence at a time (Eq. (4)):           -> mmr_select
                     s* = argmax  lambda_m*Rel(s_i) - (1-lambda_m)*max_{s_j in A} cos(v_i, v_j)
 5. Budget         bullets for section k = n_k * b_bar, b_bar = rounded mean bullets per content
                   slide of the reference deck (4 if the deck has none). If a section has
                   too few candidates ALL of them are used, a WARNING is recorded and (as the
                   proposal says) the unmet quota passes to the section with the next-highest
                   q_k.  Nothing is ever invented.                          -> draft_deck
 6. Trimming       citation brackets, citation-like / long parentheticals, and (for very long
                   sentences only) a cut at a clause boundary.  No rewriting.   -> trim_sentence
 7. Layout         the selected sentences of a section are put back into paper order and cut
                   into n_k consecutive chunks (one per planned slide).

Implementation parameters that the proposal leaves open (documented, NOT learned):
  lambda_m = 0.70, relevance weights = 0.25 each, min 6 words per candidate,
  max 35 words per bullet before a clause cut, math-heaviness threshold 4.  The proposal tunes such values on a
  development split; this phase uses neutral defaults and exposes all of them.

Determinism: scikit-learn TF-IDF + NumPy only; ties are broken by (relevance, sentence id).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from section_tagger import CANONICAL_SECTIONS

SECTIONS: List[str] = list(CANONICAL_SECTIONS)
SCHEMA = "audittailor.draft.v1"

# ---- implementation parameters (defaults; NOT learned) ------------------------------------- #
DEFAULT_LAMBDA_MMR = 0.70
DEFAULT_REL_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)  # topical, keyword, section, evidence
DEFAULT_BULLETS_PER_SLIDE = 4          # proposal: "4 if the reference deck has none"
MIN_SENTENCE_WORDS = 6
HARD_MAX_SENTENCE_WORDS = 100
DEFAULT_MAX_BULLET_WORDS = 35
MIN_CUT_WORDS = 12
WEIGHT_NAMES = ("topical", "keyword", "section", "evidence")


# =========================================================================================== #
# 1. Candidates
# =========================================================================================== #
@dataclass(frozen=True)
class Candidate:
    """A Phase 1 sentence that may become a bullet (never modified)."""
    idx: int            # Phase 1 integer id (paper order)
    sentence_id: str    # "S012"
    page: int
    section: str        # canonical label (Motivation ... Conclusion)
    heading: str        # original heading text of the section
    text: str


def format_sentence_id(i: int) -> str:
    return f"S{int(i):03d}"


def _norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


_CAPTION_RX = re.compile(r"^\s*(?:Figure|Fig\.?|Table|Algorithm|Listing)\s+\d+[A-Za-z]?\s*[:.|]", re.I)
_BIBLIO_RX = re.compile(r"\b(?:arXiv|Proc\.|Proceedings|Journal of|pp\.\s*\d+|pages?\s+\d+)\b")
_YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")
_START_OK = re.compile(r"^[A-Z0-9\"“‘'(\[]")
_LIST_ITEM_START = re.compile(r"^\(\s*(?:\d+|[ivxIVX]+|[a-zA-Z])\s*\)\s")          # "(2) A score and ..." list fragments
_MATH_STRONG = re.compile(r"[{}\u2211\u220f\u222b\u2208\u2209\u2282\u2286\u222a\u2229\u2200\u2203\u2202\u2207\u2113]")
_MATH_WEAK = re.compile(r"[=<>\u2248\u2264\u2265\u0391-\u03a9\u03b1-\u03c9]")
_END_OK = re.compile(r"[.!?][\"”’')\]]*$")


def unsuitable_reason(text: str, *, min_words: int = MIN_SENTENCE_WORDS) -> Optional[str]:
    """Return why a sentence is NOT a usable body sentence, or None if it is usable."""
    t = " ".join(text.split())
    words = t.split()
    if len(words) < min_words:
        return "too_short"
    if len(words) > HARD_MAX_SENTENCE_WORDS:
        return "too_long"
    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t.replace(" ", ""))) < 0.6:
        return "not_prose"
    if _CAPTION_RX.match(t):
        return "caption"
    # equation debris from PDF extraction: braces/sum/element-of count double, '=', '<', Greek count once
    if 2 * len(_MATH_STRONG.findall(t)) + len(_MATH_WEAK.findall(t)) >= 4:
        return "math_heavy"
    if _LIST_ITEM_START.match(t):
        return "list_item_fragment"
    if _BIBLIO_RX.search(t) and _YEAR_RX.search(t):
        return "reference_like"
    if not _START_OK.match(t):
        return "fragment_start"
    if not _END_OK.search(t):
        return "fragment_end"
    return None


def load_candidates(parsed: Mapping[str, Any], *, min_words: int = MIN_SENTENCE_WORDS) -> Tuple[List[Candidate], Dict[str, int]]:
    """Build the candidate list from a Phase 1 parsed-paper dict.

    Excluded: sentences without a canonical label, the title, abstract sentences, captions,
    reference-like lines and fragments.  Returns (candidates in paper order, {reason: count}).
    """
    abstract_n = _norm_text(parsed.get("abstract") or "")
    title_n = _norm_text(parsed.get("title") or "")
    caption_n = [_norm_text(c.get("text", "")) for c in parsed.get("captions", [])]
    excluded: Dict[str, int] = {}

    def skip(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    out: List[Candidate] = []
    for s in parsed.get("sentences", []):
        label = s.get("label")
        if label not in SECTIONS:
            skip("no_canonical_label")
            continue
        text = " ".join(str(s["text"]).split())
        n = _norm_text(text)
        if title_n and n == title_n:
            skip("title")
            continue
        if len(n) > 20 and abstract_n and n in abstract_n:
            skip("abstract")
            continue
        if len(n) > 20 and any(n in c for c in caption_n if c):
            skip("caption")
            continue
        reason = unsuitable_reason(text, min_words=min_words)
        if reason:
            skip(reason)
            continue
        out.append(Candidate(int(s["id"]), format_sentence_id(s["id"]), int(s["page"]), label,
                             str(s.get("section", "")), text))
    out.sort(key=lambda c: c.idx)
    return out, excluded


# =========================================================================================== #
# 2. TF-IDF
# =========================================================================================== #
@dataclass
class TfidfSpace:
    """TF-IDF representation of the candidate sentences (and the title+abstract query)."""
    vectorizer: TfidfVectorizer
    raw: Any            # scipy CSR, un-normalised tf-idf (used for the keyword signal)
    unit: Any           # scipy CSR, L2-normalised rows (used for every cosine)
    query_unit: Optional[Any]   # L2-normalised title+abstract vector, or None if no query text

    @property
    def n_terms(self) -> int:
        return int(self.raw.shape[1])


# Tokens must contain a letter: bare numbers ("0.87" -> "87") would otherwise become very
# high-IDF terms and blur the keyword signal with the evidence signal.
_TOKEN_PATTERN = r"(?u)\b(?=\w*[^\W\d_])\w\w+\b"


def build_tfidf(texts: Sequence[str], query_text: str = "") -> TfidfSpace:
    """Fit TF-IDF on the sentences of the paper (IDF over sentences, Appendix B)."""
    if not texts:
        raise ValueError("no candidate sentences to build TF-IDF vectors from")
    vec = TfidfVectorizer(lowercase=True, stop_words="english", token_pattern=_TOKEN_PATTERN,
                          norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
    try:
        raw = vec.fit_transform(list(texts)).tocsr()
    except ValueError as e:      # empty vocabulary
        raise ValueError(f"cannot build TF-IDF vectors: {e}") from e
    unit = normalize(raw, norm="l2", axis=1, copy=True).tocsr()
    q_unit = None
    if query_text.strip():
        q_raw = vec.transform([query_text])
        if q_raw.nnz > 0:
            q_unit = normalize(q_raw, norm="l2", axis=1, copy=True).tocsr()
    return TfidfSpace(vec, raw, unit, q_unit)


def cosine_matrix(unit_rows: Any) -> np.ndarray:
    """Dense cosine-similarity matrix of L2-normalised sparse rows."""
    sim = (unit_rows @ unit_rows.T).toarray()
    return np.clip(sim, 0.0, 1.0)


# =========================================================================================== #
# 3. Relevance
# =========================================================================================== #
_REF_NUM = re.compile(
    r"\b(?:Sections?|Secs?\.|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Appendix|Appendices|Algorithm|Stages?|"
    r"Phases?|Steps?|Lines?|Chapters?|Theorems?|Lemmas?|Definitions?)\s*\(?[A-Za-z]?\d+(?:\s*[.\u2013\-]\s*\d+)*\)?"
    r"|\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]", re.I)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*")
_RESULT_WORD = re.compile(
    r"\b(?:accuracy|accuracies|precision|recall|f1|f-?score|f-?measure|auc|bleu|rouge|scores?|errors?|loss|"
    r"perplexity|latency|throughput|runtime|speed-?up|improv\w*|outperform\w*|increas\w*|decreas\w*|reduc\w*|"
    r"gains?|gained|faster|slower|higher|lower|better|worse|achiev\w*|reach\w*|rises?|rose|risen|drops?|dropped|"
    r"significant\w*|median|mean|average|percent|percentage|fold|times)\b", re.I)
_UNIT = re.compile(r"\d\s?(?:ms|sec|seconds?|minutes?|hours?|gb|mb|kb|fps|hz|x|\u00d7)\b", re.I)
_TECH_INDICATOR = re.compile(r"[A-Za-z\u0391-\u03a9\u03b1-\u03c9][\w]*\s*(?:=|\u2248|\u2264|\u2265|<|>)\s*[-+]?\d")


def evidence_indicator(text: str) -> int:
    """Rule-based 0/1 evidence flag (Appendix B): a number together with a result word.

    Extensions kept deliberately small and listed here: '%' and measurement units count as
    result words; a 'variable = number' pattern (e.g. 'N = 10') counts as a technical
    indicator.  Numbers that are only references (Section 3, Table 2, [5], years) are ignored.
    """
    cleaned = _REF_NUM.sub(" ", text)
    nums = [m.group(0) for m in _NUMBER.finditer(cleaned)]
    nums = [n for n in nums if not re.fullmatch(r"(?:19|20)\d{2}", n.lstrip("+-"))]
    has_number = bool(nums)
    has_result = bool(_RESULT_WORD.search(cleaned)) or "%" in cleaned or bool(_UNIT.search(cleaned))
    if has_number and has_result:
        return 1
    return 1 if _TECH_INDICATOR.search(cleaned) else 0


def validate_weights(weights: Sequence[float]) -> Tuple[float, float, float, float]:
    """Four non-negative weights, normalised to sum to 1 (proposal, Appendix B)."""
    w = list(weights)
    if len(w) != 4 or any((not isinstance(x, (int, float))) or math.isnan(x) or x < 0 for x in w) or sum(w) <= 0:
        raise ValueError("relevance weights must be 4 non-negative numbers with a positive sum "
                         "(topical, keyword, section, evidence)")
    s = float(sum(w))
    return tuple(float(x) / s for x in w)  # type: ignore[return-value]


def relevance_scores(space: TfidfSpace, cands: Sequence[Candidate], q_share: Mapping[str, float],
                     weights: Sequence[float] = DEFAULT_REL_WEIGHTS) -> Dict[str, np.ndarray]:
    """Rel(s) and its four components for every candidate (all arrays aligned with ``cands``).

    topical  = cosine(v_s, v_{title+abstract})            (0 if the paper has no title/abstract text)
    keyword  = mean TF-IDF weight of the sentence's terms / max over the paper   (0 if max is 0)
    section  = q_k / max_j q_j  with q from the Phase 2 plan  (constant inside a section, so it
               never changes the ranking within one section; it matters when sentences of
               different sections compete, i.e. in the quota hand-over)
    evidence = rule-based 0/1 flag
    Rel      = weighted average with weights normalised to sum 1.
    """
    w = validate_weights(weights)
    n = len(cands)
    if space.query_unit is None:
        topical = np.zeros(n)
    else:
        topical = np.clip((space.unit @ space.query_unit.T).toarray().ravel(), 0.0, 1.0)

    nnz = np.asarray(space.raw.getnnz(axis=1), dtype=float)
    sums = np.asarray(space.raw.sum(axis=1)).ravel()
    mean_w = np.divide(sums, nnz, out=np.zeros(n), where=nnz > 0)
    keyword = mean_w / mean_w.max() if mean_w.max() > 0 else np.zeros(n)

    q_max = max((float(v) for v in q_share.values()), default=0.0)
    section = np.array([float(q_share.get(c.section, 0.0)) / q_max if q_max > 0 else 0.0 for c in cands])

    evidence = np.array([float(evidence_indicator(c.text)) for c in cands])
    rel = w[0] * topical + w[1] * keyword + w[2] * section + w[3] * evidence
    return {"topical": topical, "keyword": keyword, "section": section, "evidence": evidence,
            "rel": np.clip(rel, 0.0, 1.0)}


# =========================================================================================== #
# 4. MMR
# =========================================================================================== #
@dataclass
class Pick:
    pos: int            # position in the arrays passed to mmr_select
    rank: int           # 1 = first MMR pick of the section
    rel: float
    max_sim: float      # max cosine to the already selected sentences (0 for the first pick)
    mmr: float


def validate_lambda_m(lam: float) -> float:
    if isinstance(lam, bool) or not isinstance(lam, (int, float)) or math.isnan(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda_mmr must be in [0, 1], got {lam!r}")
    return float(lam)


def mmr_select(rel: np.ndarray, sim: np.ndarray, n_pick: int, lambda_m: float,
               tie_keys: Optional[Sequence[int]] = None,
               already: Sequence[int] = ()) -> List[Pick]:
    """Maximal Marginal Relevance, Eq. (4):

        s* = argmax_{s_i in C \\ A} [ lambda_m * Rel(s_i) - (1 - lambda_m) * max_{s_j in A} cos(v_i, v_j) ]

    ``A`` starts as ``already`` (empty -> the max term is defined as 0, so the first pick is
    the highest-relevance candidate).  Ties: higher relevance, then smaller ``tie_keys``.
    Returns the new picks in selection order (at most ``n_pick``, fewer if the pool is smaller).
    """
    lam = validate_lambda_m(lambda_m)
    n = len(rel)
    if sim.shape != (n, n):
        raise ValueError("sim must be an n x n matrix aligned with rel")
    keys = np.arange(n) if tie_keys is None else np.asarray(list(tie_keys))
    avail = np.ones(n, dtype=bool)
    max_sim = np.zeros(n)
    rank0 = len(already)
    for a in already:
        avail[a] = False
        max_sim = np.maximum(max_sim, sim[a])
    picks: List[Pick] = []
    for _ in range(max(0, min(n_pick, int(avail.sum())))):
        score = lam * rel - (1.0 - lam) * max_sim
        # lexsort: last key is primary -> score desc, then rel desc, then tie key asc
        order = np.lexsort((keys, -np.round(rel, 12), -np.round(np.where(avail, score, -np.inf), 12)))
        best = int(next(i for i in order if avail[i]))
        picks.append(Pick(best, rank0 + len(picks) + 1, float(rel[best]), float(max_sim[best]), float(score[best])))
        avail[best] = False
        max_sim = np.maximum(max_sim, sim[best])
    return picks


# =========================================================================================== #
# 5. Conservative trimming (no rewriting)
# =========================================================================================== #
_CITE_NUM = re.compile(r"\s*\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]")
_PAREN = re.compile(r"\s*\(([^()]*)\)")
_DANGLING = {"in", "by", "of", "see", "from", "to", "as", "and", "with", "ref", "refs", "reference",
             "references", "per", "following", "cf", "e.g", "i.e"}
_CLAUSE_CUT = re.compile(
    r"(?:,\s+(?:and|but|which|while|whereas|so|because|since|although|thus|therefore|where|whereby|then|yet)\b"
    r"|;\s+|:\s+|\s+\u2014\s+|\s+\u2013\s+)")
_MATHISH = re.compile(r"[=<>\u2264\u2265\u2248\u00b1\u2211\u222b\\^_]")


def _prev_word(s: str, end: int) -> str:
    m = re.search(r"([A-Za-z.]+)\s*$", s[:end])
    return m.group(1).lower().rstrip(".") if m else ""


def _is_author_year(content: str) -> bool:
    words = content.split()
    if not (_YEAR_RX.search(content) and 1 <= len(words) <= 12 and content[:1].isupper()):
        return False
    return not re.search(r"\d", _YEAR_RX.sub("", content))


def _tidy(s: str) -> str:
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return " ".join(s.split())


def trim_sentence(text: str, *, max_words: int = DEFAULT_MAX_BULLET_WORDS, min_words: int = MIN_SENTENCE_WORDS,
                  trim_parentheticals: bool = True) -> Tuple[str, List[str]]:
    """Light, reversible-in-spirit trimming.  Returns (text, operations applied).

    1. numeric citation brackets '[3]', '[5, 6]' are dropped (kept when the preceding word
       would be left dangling, e.g. 'as shown in [3]');
    2. author-year citations '(Zeng et al., 2025)' and parentheticals of >= 4 words that
       contain no digit and no math symbol are dropped (only if >= ``min_words`` words remain);
    3. only if the result is longer than ``max_words``: cut at the last clause boundary
       (', and' / ', which' / ';' / ':' / dash) that leaves 12..max_words words.  No words are
       added or changed; a final '.' is appended to the cut text.
    """
    ops: List[str] = []
    t = " ".join(text.split())

    def cite_sub(m: "re.Match[str]") -> str:
        return m.group(0) if _prev_word(m.string, m.start()) in _DANGLING else ""
    t2 = _CITE_NUM.sub(cite_sub, t)
    if t2 != t:
        ops.append("remove_citation_brackets")
        t = t2

    if trim_parentheticals:
        def par_sub(m: "re.Match[str]") -> str:
            inner = m.group(1)
            if m.start() == 0:
                return m.group(0)
            if _is_author_year(inner):
                return ""
            if len(inner.split()) >= 4 and not re.search(r"\d", inner) and not _MATHISH.search(inner):
                return ""
            return m.group(0)
        t2 = _PAREN.sub(par_sub, t)
        if t2 != t and len(_tidy(t2).split()) >= min_words:
            ops.append("remove_parenthetical")
            t = t2

    t = _tidy(t)

    if len(t.split()) > max_words:
        best = None
        for m in _CLAUSE_CUT.finditer(t):
            prefix = t[:m.start()]
            wc = len(prefix.split())
            if MIN_CUT_WORDS <= wc <= max_words and prefix.count("(") == prefix.count(")") \
                    and prefix.count("[") == prefix.count("]") \
                    and prefix.count("{") == prefix.count("}"):
                best = prefix
        if best is not None:
            best = best.rstrip(" ,;:")
            t = best if re.search(r"[.!?]$", best) else best + "."
            ops.append("clause_cut")
    return t, ops



# =========================================================================================== #
# 5b. Caption / visual matching (figures & tables already extracted in Phase 1)
# =========================================================================================== #
_PURPOSE_CAPTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "architecture": ("architecture", "pipeline", "overview", "framework", "system", "model"),
    "methodology": ("architecture", "pipeline", "overview", "method", "framework"),
    "approach": ("architecture", "pipeline", "overview", "method"),
    "main_results": ("result", "performance", "comparison", "accuracy", "table"),
    "comparison": ("comparison", "baseline", "vs", "table", "result"),
    "ablation": ("ablation", "table"),
    "analysis": ("analysis", "discussion", "table", "figure"),
    "dataset": ("dataset", "data", "corpus", "statistics", "table"),
    "experimental_setup": ("setup", "experiment", "table"),
    "metrics": ("metric", "evaluation", "table"),
}


def _match_captions(parsed: Mapping[str, Any], slide_rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Select up to 2 captions relevant to the slide purpose. Never invents captions."""
    captions = list(parsed.get("captions") or [])
    if not captions:
        return []
    purpose = (slide_rec.get("purpose") or "").lower()
    section = (slide_rec.get("section") or "").lower()
    keywords = list(_PURPOSE_CAPTION_KEYWORDS.get(purpose, ()))
    if not keywords:
        keywords = [section] if section else []
    if not keywords:
        return []
    matched: List[Dict[str, Any]] = []
    for cap in captions:
        text_c = (cap.get("text") or "").lower()
        if any(kw in text_c for kw in keywords):
            matched.append({
                "text": cap.get("text"),
                "page": cap.get("page"),
                "source": {"page": cap.get("page"), "kind": "caption"},
            })
        if len(matched) >= 2:
            break
    return matched


# =========================================================================================== #
# 6. Budget helpers
# =========================================================================================== #
def bullets_per_slide(mean_bullets: Optional[float]) -> Tuple[int, str]:
    """b_bar: mean bullets per content slide of the reference deck, rounded (half up);
    4 if the reference deck has none (proposal, Stage 4)."""
    if mean_bullets is None or isinstance(mean_bullets, bool) or not isinstance(mean_bullets, (int, float)) \
            or math.isnan(mean_bullets) or mean_bullets <= 0:
        return DEFAULT_BULLETS_PER_SLIDE, (f"reference deck has no bullet count; default b_bar = "
                                           f"{DEFAULT_BULLETS_PER_SLIDE} (proposal default)")
    b = int(math.floor(float(mean_bullets) + 0.5))
    if b < 1:
        return 1, f"reference mean {mean_bullets:.2f} rounds to 0; b_bar raised to 1 so slides are not empty"
    return b, f"b_bar = round({float(mean_bullets):.4f}) = {b}"


def split_even(items: Sequence[Any], n_parts: int) -> List[List[Any]]:
    """Cut ``items`` into ``n_parts`` consecutive chunks whose sizes differ by at most 1
    (earlier chunks get the extra item)."""
    base, rem = divmod(len(items), n_parts)
    out, pos = [], 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        out.append(list(items[pos:pos + size]))
        pos += size
    return out


# =========================================================================================== #
# 7. The drafter
# =========================================================================================== #
def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _plan_view(plan: Mapping[str, Any]) -> Tuple[int, Dict[str, int], List[str], Dict[str, float]]:
    """Read the Phase 2 plan (dict from plan.json) and check it is self-consistent."""
    try:
        alloc = {k: int(plan["slide_allocation"][k]) for k in SECTIONS}
        total = int(plan["total_slides"])
        q = {k: float(plan["blended_distribution"][k]) for k in SECTIONS}
    except KeyError as e:
        raise ValueError(f"plan JSON is missing {e}; expected a Phase 2 plan.json") from e
    if "presented_section_order" in plan:
        order = list(plan["presented_section_order"])
    else:
        order = [k for k in plan.get("section_order", SECTIONS) if alloc.get(k, 0) > 0]
    if sum(alloc.values()) != total:
        raise ValueError(f"plan is inconsistent: slide_allocation sums to {sum(alloc.values())}, total_slides = {total}")
    if sorted(order) != sorted(k for k in SECTIONS if alloc[k] > 0):
        raise ValueError("plan is inconsistent: presented_section_order does not match the sections with slides")
    return total, alloc, order, q


def draft_deck(parsed: Mapping[str, Any], qpp: Mapping[str, Any], plan: Mapping[str, Any], *,
               lambda_m: float = DEFAULT_LAMBDA_MMR,
               rel_weights: Sequence[float] = DEFAULT_REL_WEIGHTS,
               max_bullet_words: int = DEFAULT_MAX_BULLET_WORDS,
               min_sentence_words: int = MIN_SENTENCE_WORDS,
               handover: bool = True,
               inputs: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Draft the whole deck as JSON (no PPTX).  See the module docstring for the algorithm."""
    lam = validate_lambda_m(lambda_m)
    w = validate_weights(rel_weights)
    if max_bullet_words < MIN_CUT_WORDS:
        raise ValueError(f"max_bullet_words must be >= {MIN_CUT_WORDS}")
    total, alloc, order, q = _plan_view(plan)
    b_bar, b_note = bullets_per_slide(qpp.get("mean_bullets"))

    cands, excluded = load_candidates(parsed, min_words=min_sentence_words)
    if not cands:
        raise ValueError("no usable candidate sentences in the parsed paper")
    warnings: List[str] = []
    notes: List[str] = [b_note]

    query = f"{parsed.get('title') or ''} {parsed.get('abstract') or ''}".strip()
    if not query:
        warnings.append("WARNING: the parsed paper has no title/abstract; the topical-similarity signal is 0 for every sentence.")
    space = build_tfidf([c.text for c in cands], query)
    if query and space.query_unit is None:
        warnings.append("WARNING: title/abstract shares no vocabulary with the paper; the topical-similarity signal is 0.")
    comp = relevance_scores(space, cands, q, w)
    rel = comp["rel"]

    by_section: Dict[str, List[int]] = {k: [] for k in SECTIONS}
    for i, c in enumerate(cands):
        by_section[c.section].append(i)
    sim_cache: Dict[str, np.ndarray] = {}

    def sim_for(sec: str) -> np.ndarray:
        if sec not in sim_cache:
            sim_cache[sec] = cosine_matrix(space.unit[by_section[sec], :]) if by_section[sec] else np.zeros((0, 0))
        return sim_cache[sec]

    quota = {k: alloc[k] * b_bar for k in order}
    picks: Dict[str, List[Tuple[int, Pick, Optional[str]]]] = {k: [] for k in order}   # (candidate idx, pick, handover donor)

    def run_mmr(sec: str, n: int, donor: Optional[str]) -> int:
        pool = by_section[sec]
        if not pool or n <= 0:
            return 0
        already = [pool.index(ci) for ci, _, _ in picks[sec]]
        new = mmr_select(rel[pool], sim_for(sec), n, lam, tie_keys=[cands[i].idx for i in pool], already=already)
        for p in new:
            picks[sec].append((pool[p.pos], p, donor))
        return len(new)

    # ---- pass 1: each section fills its own quota from its own candidates ----------------- #
    deficit: Dict[str, int] = {}
    for k in order:
        got = run_mmr(k, quota[k], None)
        if got < quota[k]:
            deficit[k] = quota[k] - got
            warnings.append(f"WARNING: {k} section requested {quota[k]} bullets but only {len(by_section[k])} "
                            f"suitable candidates were available.")

    # ---- pass 2: proposal rule - unmet quota passes to the section with the next-highest q_k #
    received: Dict[str, int] = {k: 0 for k in order}
    given: Dict[str, int] = {k: 0 for k in order}
    if handover:
        for donor in [k for k in order if k in deficit]:
            need = deficit[donor]
            recipients = sorted((r for r in order if r != donor),
                                key=lambda r: (-q[r], SECTIONS.index(r)))
            for r in recipients:
                if need <= 0:
                    break
                got = run_mmr(r, need, donor)
                if got:
                    need -= got
                    received[r] += got
                    given[donor] += got
                    notes.append(f"NOTE: {got} unmet bullet(s) from {donor} passed to {r} "
                                 f"(next-highest q_k = {q[r]:.3f}).")
            deficit[donor] = need
            if need > 0:
                warnings.append(f"WARNING: {need} bullet(s) requested for {donor} could not be reassigned "
                                f"(no other presented section has unused candidates); the deck has fewer bullets than budgeted.")
    else:
        for donor, need in deficit.items():
            notes.append(f"NOTE: hand-over disabled; {need} requested bullet(s) for {donor} left unfilled.")

    # ---- layout: paper order inside a section, n_k consecutive chunks -------------------- #
    # Prefer purpose-driven story plan when the planner provided one.
    story = list(plan.get("story") or [])
    story_by_section: Dict[str, List[Dict[str, Any]]] = {k: [] for k in order}
    for slot in story:
        sec = slot.get("canonical_section")
        if sec in story_by_section:
            story_by_section[sec].append(slot)

    slides: List[Dict[str, Any]] = []
    n_trimmed = 0
    for k in order:
        chosen = sorted(picks[k], key=lambda t: cands[t[0]].idx)
        slots = story_by_section.get(k) or []
        if slots and any(s.get("prefers_evidence") for s in slots):
            def _evidence_key(t, _k=k):
                ci = t[0]
                ev = float(comp.get("evidence", [0.0] * len(cands))[ci]) if "evidence" in comp else 0.0
                return (-ev, cands[ci].idx)
            chosen = sorted(picks[k], key=_evidence_key)

        chunks = split_even(chosen, alloc[k])
        for j, chunk in enumerate(chunks, start=1):
            bullets = []
            for ci, p, donor in chunk:
                c = cands[ci]
                text, ops = trim_sentence(c.text, max_words=max_bullet_words, min_words=min_sentence_words)
                n_trimmed += bool(ops)
                bullets.append({
                    "text": text,
                    "source": {"sentence_id": c.sentence_id, "page": c.page, "section": c.section, "heading": c.heading},
                    "original_text": c.text,
                    "trim_ops": ops,
                    "selection": {
                        "mmr_rank": p.rank, "relevance": _r(p.rel), "max_similarity_to_selected": _r(p.max_sim),
                        "mmr_score": _r(p.mmr),
                        "relevance_components": {n: _r(comp[n][ci]) for n in WEIGHT_NAMES},
                        "from_handover_of": donor,
                    },
                })
            slide_rec: Dict[str, Any] = {
                "slide_number": len(slides) + 1,
                "section": k,
                "section_slide_index": j,
                "section_slide_count": alloc[k],
                "bullets": bullets,
            }
            if j - 1 < len(slots):
                slot = slots[j - 1]
                slide_rec["purpose"] = slot.get("purpose")
                slide_rec["title"] = slot.get("title") or k
                slide_rec["source_headings"] = list(slot.get("source_headings") or [])
                slide_rec["prefers_evidence"] = bool(slot.get("prefers_evidence"))
            else:
                slide_rec["purpose"] = None
                slide_rec["title"] = k
            captions = _match_captions(parsed, slide_rec)
            if captions:
                slide_rec["visuals"] = captions
            slides.append(slide_rec)
            if not bullets:
                warnings.append(f"WARNING: slide {slides[-1]['slide_number']} ({k}) has no bullets; nothing was invented to fill it.")

    summary = {k: {"slides": alloc[k], "requested_bullets": quota[k], "candidates": len(by_section[k]),
                   "selected": len(picks[k]), "received_via_handover": received[k], "given_via_handover": given[k]}
               for k in order}
    return {
        "schema": SCHEMA,
        "mode": "extractive (TF-IDF + MMR, key-free: no LLM, no API)",
        "inputs": dict(inputs or {}),
        "title": parsed.get("title"),
        "parameters": {
            "lambda_mmr": lam,
            "lambda_mmr_note": "implementation parameter (neutral default), not learned",
            "relevance_weights": {n: _r(x) for n, x in zip(WEIGHT_NAMES, w)},
            "reference_mean_bullets": qpp.get("mean_bullets"),
            "bullets_per_slide_b_bar": b_bar,
            "quota_handover": bool(handover),
            "max_bullet_words": max_bullet_words,
            "min_sentence_words": min_sentence_words,
            "tfidf": {"scope": "candidate sentences of the paper (IDF over sentences)", "stop_words": "english",
                      "n_terms": space.n_terms, "n_candidates": len(cands)},
        },
        "n_slides": len(slides),
        "section_order": list(order),
        "slide_allocation": {k: alloc[k] for k in order},
        "candidate_filter": {"parsed_sentences": len(parsed.get("sentences", [])), "kept": len(cands),
                             "excluded": dict(sorted(excluded.items()))},
        "section_summary": summary,
        "n_bullets": sum(len(s["bullets"]) for s in slides),
        "n_bullets_trimmed": n_trimmed,
        "slides": slides,
        "warnings": warnings,
        "notes": notes,
    }

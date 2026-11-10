# AuditTailor — Phase 1 (PDF parsing + QPP), Phase 2 (budgeted planner) and Phase 3 (key-free extractive drafting)

Deterministic implementation of Stages 1–2 of the AuditTailor proposal (Sec. 4.2):
parse a paper PDF into sentences with section labels and source pointers, and turn a
reference `.pptx` deck into the Quantified Preference Profile θ = (w, π, φ).

No LLM, no API key, no network, no training.

**Implemented:** Phases 1–6 end-to-end (parse → plan → extractive draft → PPTX generation → verification → repair).
Deterministic, key-free, offline. No LLM or external API at any stage.

## Install
```bash
pip install -r requirements.txt
```
(Python 3.9+; 3.10+ recommended.)

## Run
```bash
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx
```
`--paper` and `--deck` may be used independently; `--out DIR` changes the output directory
(default `outputs/phase1/`). Output files: `<paper-stem>_parsed.json`, `<deck-stem>_qpp.json`.

Generate the synthetic sample data (already included): `python tests/make_synthetic_data.py`

## Phase 2 — run
```bash
# Phase 1 + planner in one command
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx --plan --n-slides 10 --lambda 0.6

# planner only, from the Phase 1 JSON files
python src/main.py --parsed outputs/phase1/synthetic_paper_parsed.json --qpp outputs/phase1/synthetic_reference_qpp.json --plan --n-slides 10 --lambda 0.6
```
Writes `outputs/phase2/plan.json` (`--plan-out PATH` to change). `--n-slides N` is required with `--plan`; `--lambda` defaults to 0.6 (a demonstration value — the proposal tunes λ on a development split). Invalid λ or N exits with code 1 and an error message.

## Test
```bash
python tests/test_phase1.py
python tests/test_phase2.py
python tests/test_phase3.py
```
Builds a synthetic PDF + PPTX in a temp dir, runs `main.py`, and asserts on the console output
and on the JSON (section labels, w, π, β, ρ, ε, b̄, hyphen repair, header/caption/reference removal).
The synthetic deck deliberately contains a no-bullet paragraph inside a body placeholder, a prose-only slide and a chart-only slide, so the tests check that b̄ ≠ mean paragraphs, β counts prose, and ρ ignores charts.

## Files
| File | Role |
|---|---|
| `src/section_tagger.py` | Keyword table → 6 canonical labels; priority order (Results before generic Setup); exclusion of outline/reference/closing titles; inheritance rule |
| `src/parser.py` | PyMuPDF parsing: headings (bold/larger type + numbered patterns), header/footer/caption/reference removal, de-hyphenation, rule-based sentence splitter, pointers `(page, section, sentence id)` |
| `src/qpp.py` | θ = (w, π, φ) and mean bullets from a PPTX; text-fill estimate with Pillow metrics |
| `src/planner.py` | **Phase 2.** Budgeted planner: p_k, w′_k, q_k (Eq. 3), largest-remainder rounding, section order. Standard library only, exact `Fraction` arithmetic |
| `src/extractor.py` | **Phase 3.** Candidate filtering, TF-IDF (scikit-learn), relevance Rel(s), MMR (Eq. 4), bullet budget, quota hand-over, conservative trimming, JSON draft |
| `src/main.py` | CLI, JSON writers, console reports (Phase 1 always; Phase 2 with `--plan`; Phase 3 with `--draft`) |
| `tests/` | Synthetic data generator, `test_phase1.py`, `test_phase2.py`, `test_phase3.py` |

## QPP definitions as implemented
- **w**: slides per canonical section ÷ content slides (title, outline, references, closing, section-divider slides excluded; a title with no keyword match inherits the previous slide's label).
- **π**: labels in order of first appearance.
- **η**: mean over all non-empty text boxes on content slides of (estimated text height ÷ inner box height). Raw mean, not capped at 1.
- **β**: fraction of content slides whose body (non-title) text has ≥ 3 non-empty paragraphs. Paragraphs of **any** kind count (prose or bullet); β is *not* a bullet metric.
- **ρ**: fraction of content slides with ≥ 1 **picture or table** (strict proposal definition). Charts do **not** count; they are recorded only as a diagnostic (`n_charts` per slide in the JSON) and never affect ρ.
- **ε**: fraction of content slides with an equation: OMML math object, LaTeX markers, math symbols (∑ ∫ ≤ ≈ ∈ …) or an `x = a + b` pattern. Heuristic.
- **b̄**: mean number of **bulleted** paragraphs per content slide = (total bullet paragraphs over content slides) ÷ (number of content slides). A paragraph counts as a bullet only if PowerPoint's paragraph properties say so, resolved in this order: the paragraph's own `<a:pPr>` (`buChar`/`buAutoNum`/`buBlip` = bullet, `buNone` = not a bullet) → the shape's list style → the layout placeholder → the master placeholder → the master text styles. Text with no explicit bullet setting anywhere (e.g. a plain text box) is prose, not a bullet. Title/subtitle/footer text and table cells are not counted. b̄ is not part of the style distance. The JSON also reports `total_bullet_paragraphs` and, as a clearly labelled diagnostic, `mean_body_paragraphs` (prose + bullets), which is *not* b̄.

## Phase 2 — how it maps to the proposal (Sec. 4.3)
| Proposal | `planner.py` |
|---|---|
| ℓ_k, P = {k : ℓ_k > 0} | `label_word_counts_l` from Phase 1; `present = [k for k in SECTIONS if words[k] > 0]` |
| p_k = ℓ_k / Σ_{j∈P} ℓ_j | `paper_distribution` |
| w′_k = w_k / Σ_{j∈P} w_j; if the sum is 0, w′_k = p_k | `restrict_and_renormalise` (fallback recorded in `fallbacks`) |
| Eq. (3) q_k = λ w′_k + (1−λ) p_k | `blend`; λ validated to a finite number in [0, 1] by `validate_lambda` |
| ⌊N q_k⌋ plus one extra slide for each of the m = N − Σ⌊N q_j⌋ largest fractional remainders, ties to the larger q_k | `largest_remainder` (remaining ties → canonical order, so results are deterministic) |
| n_k = 0 ⇒ section not presented | allowed; a warning is emitted, no minimum is forced |
| Order follows π, sections not in P skipped, sections of P missing from π inserted before Conclusion | `order_sections` (if Conclusion is not in the order, they are appended; a missing Conclusion goes last) |

Exact rational arithmetic (`fractions.Fraction`, floats read via their shortest decimal) is used throughout, so ties are decided by the numbers and not by float noise (`10 * 0.18 == 1.7999999999999998` in floats). `plan.json` contains the paper / reference / restricted-reference / blended distributions, the integer allocation, the section order, and a per-section audit (N·q, floor, remainder, extra slide). Nothing in Phase 2 measures preference distance (D_pref/PF) — that belongs to the verification stage.

## Phase 3 — key-free extractive drafting (Sec. 4.4, Eq. (4), Appendix B/C)
```bash
# from the Phase 1 / Phase 2 JSON files only (no PDF needed)
python src/main.py --parsed outputs/phase1/synthetic_paper_parsed.json --qpp outputs/phase1/synthetic_reference_qpp.json --plan-json outputs/phase2/plan.json --draft

# all three phases in one command
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx --plan --n-slides 10 --draft
```
Writes `outputs/phase3/draft.json` (`--draft-out PATH`). **No LLM, no API, no key, no training; no PPTX yet.** Every bullet is a sentence of the paper (optionally trimmed) with a pointer `(sentence_id, page, section)`.

Options: `--lambda-mmr` (default 0.70), `--rel-weights TOPICAL KEYWORD SECTION EVIDENCE` (default 0.25 each, normalised), `--max-bullet-words` (35), `--no-handover`. These are **implementation parameters, not learned values**; the proposal tunes them on a development split.

How it maps to the proposal
- **Candidates** `C_k`: Phase 1 sentences with label *k*, minus title, abstract, captions, reference-like lines, fragments, list-item fragments and equation debris (`math_heavy`). Counts per exclusion reason are in `candidate_filter` of the JSON.
- **TF-IDF**: scikit-learn, IDF over the candidate sentences (each sentence is a document), English stop words removed, tokens must contain a letter (bare numbers are not terms).
- **Rel(s)** = weighted average of: topical (cosine to title + abstract), keyword (mean TF-IDF weight of the sentence's terms ÷ the paper maximum), section (q_k ÷ max q_j from the Phase 2 plan), evidence (0/1: a number together with a result word, `%` or a unit; numbers that are only references such as *Section 3*, *Table 2*, `[5]` or years do not count; `N = 10`-style indicators count as technical evidence). The section signal is constant inside a section, so — as the proposal notes — it never changes the within-section ranking; it only matters for the quota hand-over.
- **MMR**: `s* = argmax λm·Rel(s_i) − (1−λm)·max_{s_j∈A} cos(v_i, v_j)`, A per section, first pick = highest relevance, ties → higher Rel, then earlier sentence.
- **Budget**: bullets for section *k* = n_k × b̄, b̄ = round(`mean_bullets` of the QPP) (half up; 4 if the reference deck has none). If a section has too few candidates, all are used, `WARNING: <Section> section requested N bullets but only M suitable candidates were available.` is recorded and — as stated in the proposal — the unmet quota passes to the section with the next-highest q_k (`--no-handover` disables this; nothing is ever invented either way).
- **Layout**: selected sentences of a section go back into paper order and are cut into n_k consecutive chunks (one per planned slide). Slide count and section order are exactly Phase 2's `presented_section_order` / `slide_allocation`.
- **Trimming** (no rewriting): drop numeric citation brackets (`[3]`, `[5, 6]`; kept if the preceding word would dangle, e.g. *shown in [3]*), author–year citations, parentheticals of ≥ 4 words without digits/math, and — only for sentences over `--max-bullet-words` — a cut at the last clause boundary (`, and` / `, which` / `;` / `:` / dash) that leaves 12..max words. The untouched sentence is kept as `original_text`, the operations as `trim_ops`.

Phase 3 limitations: bullets are paper prose (no abstraction); section labels come from Phase 1, so a mis-tagged section sends sentences to the wrong slide; the evidence indicator is lexical; the PDF parser can leave equation debris that the `math_heavy` filter only partly catches; trimming can drop information (e.g. a clause after a colon); no grounding / overflow checks yet (Phase 4+).

## Limitations / known issues
- **Section tagging is keyword-based.** Ambiguous headings follow the table (e.g. "Evaluation and Limitations" → Results; "Attribution, Extractive Selection and Evaluation" → Results). Tagging accuracy against hand labels is not measured yet (planned in the proposal, Sec. 4.6).
- **Heading detection** relies on bold/larger type and strictly sequential numbering plus a font-size consistency check. A paper with a heading that is missed (or non-bold, unnumbered headings) can cause later numbered headings to be rejected, merging sections. Headings must fit on one line (≤ 90 chars).
- **Bold run-in headings** (e.g. "Goal:", "The idea in brief.") are not sections; very short ones are dropped as fragments, but a run-in heading of ≥ 5 words ending in a period can survive as a "sentence".
- Everything after the References heading (including appendices) is excluded. Abstract is stored separately, not as sentences.
- Two-column layouts use a simple left-then-right ordering heuristic and were **not** tested on a real two-column paper. Scanned PDFs (no text layer) are rejected; no OCR.
- Hyphen repair joins "extrac-"+"tive" but also turns a true compound broken at a line end ("well-"+"known") into "wellknown".
- Prose filter drops fragments (figure/table text, equations) using word-count, letter-ratio and terminal-punctuation rules, so some genuine short sentences are dropped (counted in `diagnostics`).
- **η is an estimate**: it uses Liberation Sans / Arial / DejaVu metrics (not the deck's real font), a 1.2 line height, and approximate bullet indents. If no TrueType font is found it falls back to a constant 0.5 em width (a warning is emitted). `spAutoFit` boxes are measured like fixed boxes. It does not replace rendering in PowerPoint.
- Equation images (pictures of equations) are not detected as equations.
- **Bullet detection** reads the bullet settings stored in the file. A deck that fakes bullets by typing a "•" or "-" character into the text is counted as prose, and a deck whose template/master defines no bullet properties for a body placeholder gets 0 bullets. Bullets inside tables and in grouped shapes' inherited styles are not examined.
- Charts are detected (`n_charts`) but excluded from ρ; a deck that presents all its figures as native charts will therefore have ρ = 0.
- Only tested on synthetic data and on the AuditTailor proposal PDF; no real reference `.pptx` from a human presenter has been tested.
- **Phase 2 inherits Phase 1's errors.** ℓ_k comes from Phase 1's keyword section tagging and sentence filter (fragments dropped there do not count as words), and w/π come from the QPP; a mis-tagged heading or slide shifts the plan. With the 6-label scheme, a paper whose experiments section is tagged only Setup or only Results will have the other label absent.
- λ is a free parameter; the 0.6 default is for demonstration, not a tuned value. The plan is a slide *budget* only: nothing yet checks that a section's text can fill its slides, and no slide content is produced.
- With N smaller than the number of present sections, some sections get 0 slides by design (warned in the JSON/console).
- Reference weights are read from the QPP JSON, which stores them rounded to 6 decimals; ties in the rounding step are resolved on those stored values.

## Phase 4 – editable PPTX generation (template filling)

**What it does.** `src/pptx_generator.py` turns the Phase 2 plan and the Phase 3 draft into an editable `.pptx`
using python-pptx. It opens the template deck, keeps its slide size, master, theme and layouts, removes the
template's example slides, and creates a title slide, one content slide per planned slide (template
"Title and Content" layout) and a final *Source Index* slide (`--no-source-slide` to omit). All text is real,
editable placeholder text taken verbatim from the draft: no LLM, no paraphrasing, no padding. The planner's section
order and per-section slide counts are used as-is; a draft that disagrees with the plan is an error. A planned slide
with no bullets is skipped with a warning rather than emitted empty.

**Inputs.** `outputs/phase2/plan.json`, `outputs/phase3/draft.json`, a template PPTX
(default: the `--deck` reference deck, else `data/reference_decks/synthetic_reference.pptx`).

**Outputs.** `outputs/phase4/audittailor_generated.pptx` and `outputs/phase4/pptx_metadata.json`
(slide-to-section map, bullets per slide, source-pointer count, overflow checks/adjustments, warnings, errors).

**Run.**
```
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx \
                   --plan --n-slides 10 --draft --pptx                    # Phases 1-4
python src/main.py --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json --pptx   # Phase 4 only
python tests/test_phase4.py
```
Options: `--template`, `--pptx-out`, `--min-font-size` (default `MIN_FONT_SIZE = 16`), `--no-source-slide`.

**Overflow.** The fill of each body box is estimated with the Phase 1 text-fit estimator. Above `MAX_FILL` the font is
reduced in 2 pt steps down to the minimum, then paragraph spacing is tightened. Text is never deleted or truncated;
adjustments and any box still too full at the minimum are recorded in the metadata for Phase 5. The estimate is
heuristic (no PowerPoint rendering).

**Traceability.** Each bullet ends with a small grey marker in the same paragraph, e.g. `[S004, p.1]` (Phase 3
sentence id and page). The speaker notes of each content slide give the full pointer (id, page, section, original
heading), and the Source Index slide lists all markers per slide. Pointers are copied from the draft, never invented.

## Phase 5 – verification and evaluation

**What is verified.** `src/verifier.py` inspects the generated PPTX (Phase 4 metadata is not trusted) against the Phase 2
plan, the reference QPP and the parsed paper. It only *measures*; repair is Phase 6. All metrics are in [0, 1],
deterministic, key-free and offline. Defaults for the weights are implementation choices, not learned values.

| Metric | Definition |
|---|---|
| **Dalloc** | `(1/K) Σ_k min(1, abs(planned_k − actual_k) / max(1, planned_k))` over sections with planned or actual slides > 0 (K = their number). `actual` = content slides per section of the generated deck, counted with the Phase 1 QPP extractor. |
| **Dorder** | Normalised Kendall inversion distance between the plan's section order and the generated deck's order π. Pairs that cannot be ranked (section missing from one order) count as disagreements. 0 = identical, 1 = reversed/disjoint. |
| **Dstyle** | Mean absolute difference (each clamped to 1) of φ = (η, β, ρ, ε) between reference and generated QPP. Only these four QPP features are used; w and π are covered by Dalloc/Dorder, and colour/font/layout are future work. |
| **Dpref** | `λ1·Dalloc + λ2·Dorder + λ3·Dstyle`, λ sum to 1 (default 0.4 / 0.2 / 0.4, `--dpref-weights`). |
| **PF** | `1 − Dpref`. |
| **SR** | Fraction of bullets whose source pointer resolves to a paper sentence (same page) and whose TF-IDF cosine with that sentence is ≥ threshold (default 0.70, `--sr-threshold`). |
| **NF** | Correct / checked numeric claims: every number in a bullet must occur (same value) in its source sentence. |
| **O** | Overflowed / checked text boxes, measured on the PPTX: estimated text height > box height, or a text run below the minimum font size (source markers excepted). |
| **J(S)** | `Dpref + μ·O + ν·(1 − NF) + ξ·(1 − SR)`; lower is better; default μ = 0.5, ν = ξ = 1 (`--objective-weights`). |

**SR and NF are heuristics.** SR is lexical overlap, not semantic entailment: a reworded or negated bullet can score
high or low regardless of truth. NF is a regex check: number words ("five"), units and percent signs are not parsed, and
it does not check that a number is attached to the right quantity. Both are meaningful for the current extractive
Phase 3, where bullets are source sentences or clause-trimmed subsets. The overflow test uses the Phase 1 text-fit
estimate, not a PowerPoint render.

**Notes.** The Phase 4 "Source Index" slide is excluded from the generated-deck QPP (it is reference-only, and Phase 1
would otherwise count it as a content slide). An extractive deck has no pictures, tables or equations, so ρ and ε
differences from a reference that has them are expected and show up in Dstyle.

**Output / run.** `outputs/phase5/verification.json` holds the metrics at top level (`dalloc`, `dorder`, `dstyle`,
`dpref`, `pf`, `sr`, `nf`, `overflow_rate`, `objective`) plus `config`, `details` (per-section terms, discordant
pairs, per-bullet cosine and unsupported ids, incorrect numbers, per-text-box fill/affected slides) and `warnings`.
```
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx \
                   --plan --n-slides 10 --draft --pptx --verify                  # Phases 1-5
python src/main.py --parsed outputs/phase1/synthetic_paper_parsed.json --qpp outputs/phase1/synthetic_reference_qpp.json \
                   --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json \
                   --generated-pptx outputs/phase4/audittailor_generated.pptx --verify   # Phase 5 only
python tests/test_phase5.py
```

## Phase 6 – bounded repair loop

`src/repair.py` repairs a generated deck using only what already exists: the Phase 3 draft, the parsed source
sentences, the Phase 2 plan and Phase 4's own overflow fitter. **The repair is fully deterministic. It does not use an
LLM, an API or any external service, and it never writes new scientific text.**

**Loop.** `verify → criteria met? → repair → verify again → keep the lowest-J(S) candidate`. One round repairs every
failing family, in the proposal's priority order, then re-fits any box that the restored content made overflow:

| Priority | Family | Trigger (all configurable in `repair.py`) | What it does |
|---|---|---|---|
| 1 | **Overflow** | `O > MAX_OVERFLOW_RATE` (0.0) | Phase 4's `fit_body` on the flagged boxes: font steps down to the minimum (runs below the minimum are raised to it first), then tighter spacing. Text is never deleted or shortened. |
| 2 | **Support / fidelity** | `SR < MIN_SUPPORT_RATIO` (1.0) or `NF < MIN_NUMERIC_FIDELITY` (1.0) | Each failing bullet is replaced by the first candidate that passes the verifier's own checks: its Phase 3 draft text → its original source sentence → the parsed sentence at its pointer. The source marker `[Sxxx, p.N]` is preserved/restored. Numbers come back with the source text; they are never guessed. |
| 3 | **Preference drift** | `Dalloc > MAX_DALLOC` (0), `Dorder > MAX_DORDER` (0) or `Dpref > MAX_PREFERENCE_DISTANCE` (0.15) | Slides are matched to the plan via their source pointers; wrong section labels are retitled and slides reordered in place (Source Index and notes refreshed). Missing/extra/duplicate slides trigger a rebuild from plan + draft with the Phase 4 generator. The plan stays authoritative; the reference QPP is never touched. |

**Rounds.** `MAX_REPAIR_ROUNDS = 3` (`--max-repair-rounds`); the loop can never run longer. It stops early when all
criteria hold, when no unblocked failing family remains, or when a round changes nothing.

**Objective and best candidate.** `J(S) = Dpref + μ·O + ν·(1 − NF) + ξ·(1 − SR)` (Phase 5 weights). A round's candidate
becomes the best only if its verified J strictly decreases. A non-improving candidate is recorded, rejected, the loop
returns to the best candidate and blocks the repair families that were tried. The final output is always the
candidate with the lowest verified J, never just the last one. A deck that already satisfies all criteria is left
byte-identical (0 rounds).

**Not repairable.** Dstyle (η, β, ρ, ε vs the reference) is not repaired, because that would mean changing content. If
Dpref exceeds its threshold only because of Dstyle, the loop makes no change and stops (`no_effective_repair`).

**Outputs** (`outputs/phase6/`, `--repair-out DIR`): `final.pptx` (editable), `final_verification.json` (full Phase 5
result of the final candidate), `repair_history.json`, `final_draft.json`, and `candidates/round_N.pptx` (only for
rounds that changed something). `repair_history.json` records, per round: detected failures, each action with **why**
(failed metric), **what** (old/new text, retitled/moved slides, font changes), **source** (draft / source sentence /
plan), affected slides and bullets, previous J, new J, whether it improved and whether it became the best; plus the
initial and final metrics and the stop reason.

```
python src/main.py --paper data/papers/synthetic_paper.pdf --deck data/reference_decks/synthetic_reference.pptx \
                   --plan --n-slides 10 --draft --pptx --repair                 # Phases 1-6 (Phase 5 runs first)
python src/main.py --parsed outputs/phase1/synthetic_paper_parsed.json --qpp outputs/phase1/synthetic_reference_qpp.json \
                   --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json \
                   --generated-pptx outputs/phase4/audittailor_generated.pptx --repair   # Phase 6 on an existing deck
python tests/test_phase6.py
```
Threshold flags: `--max-overflow-rate`, `--min-support`, `--min-nf`, `--max-pref-distance`, `--max-repair-rounds`.
The Phase 6 tests tamper with the real deck (overflow, unsupported text, altered numbers, wrong labels/order/missing
slide) and also run Phases 1–5 as a regression check.

## Run the Web Interface

No Python, file paths or command-line options are needed to *use* AuditTailor; you only start it once.

1. Install the dependencies (once): `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`
3. Open the local address Streamlit prints in the terminal (usually http://localhost:8501).

> **Built-in styles are bundled reference presentations used to provide presentation preferences. Users do not need to
> find a reference presentation themselves.**

### Simple workflow (only a PDF is needed)
1. Upload the **research paper (PDF)**.
2. Select a **Presentation Style** (default: *Academic Research*).
3. Click **Generate Presentation**.
4. Click **Download PowerPoint** (the verification report can be downloaded too).

### Advanced workflow (your own reference presentation)
1. Upload the **research paper (PDF)**.
2. Tick **Use my own reference presentation**.
3. Upload your **reference PPTX** (a presentation whose structure and presentation style you want AuditTailor to follow;
   it is used *instead of* the selected built-in style).
4. Optionally open **Advanced Settings** (number of content slides, maximum repair rounds, minimum support ratio,
   minimum numeric fidelity, maximum overflow rate; the defaults are the project's standard values).
5. Click **Generate Presentation**.
6. Download the **PowerPoint** and the **verification report** (JSON); **Repair Details** shows the repair rounds and the
   repair history can be downloaded as well.

After generation the page states which reference was used (e.g. *Reference style: Academic Research (built-in)* or
*Reference style: Custom uploaded presentation*) and shows Dalloc, Dorder, Dstyle, Dpref, PF, SR, NF, O and J(S), all read
from the generated `final_verification.json` / `repair_history.json` (nothing is hard-coded).

### Built-in styles
| Style | Idea (structure only) |
|---|---|
| Academic Research (default) | Balanced sections, concise academic bullets |
| Technical / AI | More slides on method and results, more equations/figures |
| Conference | Compact, stronger results emphasis |
| Minimal Professional | Fewer bullets, shorter text, more whitespace |
| Detailed Research | More slides and more bullets per section |

The styles are **not scientifically validated**: each is just a small, generic reference deck in
`data/reference_decks/builtin/` (no scientific content; regenerate with `python tools/build_builtin_decks.py`).
Selecting a style does **not** bypass the pipeline. It only chooses the reference file:

```
paper PDF + (built-in style -> built-in reference PPTX | your own reference PPTX)
  -> Phase 1 QPP extraction -> Phase 2 planning -> Phase 3 extractive drafting
  -> Phase 4 PPTX generation -> Phase 5 verification -> Phase 6 bounded repair -> final PPTX
```

*Reference vs. template.* The reference presentation teaches the **preferences** (section allocation, order, fill ratio,
bullet/paragraph density, picture/table and equation usage). The **visual template** (master, layouts, slide size) is a
separate role; as before, it defaults to the reference deck itself (so the built-in decks are valid templates too), but
`pipeline.run_pipeline(..., template_pptx=...)` can take a different file. A purely visual template with generic
placeholder text (for example a downloaded "AI template") should be used as a *template*, not as a reference
presentation, because its slide structure says nothing about how the user presents a paper.

Friendly checks: the PDF must be a readable PDF with selectable text (scanned PDFs are not supported), the reference a
valid PPTX with at least one content slide, and a missing built-in file is reported with a clear message. Technical details
(stack traces) are hidden unless `AUDITTAILOR_DEBUG=1` is set.

How it is built: `app.py` is only the interface (collect inputs, call `src/pipeline.py`, show results, offer downloads).
`src/pipeline.py` (`stage_inputs`, `run_pipeline`) runs the existing Phase 1–6 modules unchanged; `src/builtin_styles.py`
is the style registry. Uploads are validated (extension, file signature, readability), saved under generated names in a
temporary directory, only read as data, and deleted afterwards; the project's `data/` and `outputs/` folders are not
touched. The pipeline is local, key-free and deterministic: no login, database, external service or LLM.

Tests: `python tests/test_app.py` (interface/pipeline layer + Phase 1–6 regression) and
`python tests/test_ui_reference.py` (styles, custom reference, validation, end-to-end runs, the Streamlit page).

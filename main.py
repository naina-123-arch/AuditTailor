"""
main.py -- AuditTailor Phase 1 command line entry point.

    python src/main.py --paper data/papers/paper.pdf --deck data/reference_decks/ref.pptx

Writes JSON to outputs/phase1/ and prints a console report.  Either --paper or
--deck may be given alone.  Fully deterministic; no API keys, no network.

Phase 2 (budgeted planner) is switched on with --plan --n-slides N [--lambda L]:

    python src/main.py --paper P.pdf --deck D.pptx --plan --n-slides 10 --lambda 0.6
    python src/main.py --parsed outputs/phase1/P_parsed.json --qpp outputs/phase1/D_qpp.json \
                       --plan --n-slides 10 --lambda 0.6

It writes outputs/phase2/plan.json.  Without --plan nothing from Phase 2 runs.

Phase 3 (key-free extractive drafting, TF-IDF + MMR) is switched on with --draft:

    python src/main.py --parsed outputs/phase1/P_parsed.json --qpp outputs/phase1/D_qpp.json \\
                       --plan-json outputs/phase2/plan.json --draft
    python src/main.py --paper P.pdf --deck D.pptx --plan --n-slides 10 --draft     # all three phases

It writes outputs/phase3/draft.json.  No LLM, no API, no key.

Phase 4 (editable PPTX generation, template filling) is switched on with --pptx:

    python src/main.py --paper P.pdf --deck D.pptx --plan --n-slides 10 --draft --pptx     # all four phases
    python src/main.py --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json --pptx \\
                       --template data/reference_decks/synthetic_reference.pptx

It writes outputs/phase4/audittailor_generated.pptx and outputs/phase4/pptx_metadata.json.

Phase 5 (deterministic verification: Dalloc, Dorder, Dstyle, Dpref, PF, SR, NF, O, J) is switched on with --verify:

    python src/main.py --paper P.pdf --deck D.pptx --plan --n-slides 10 --draft --pptx --verify   # all five phases
    python src/main.py --parsed outputs/phase1/P_parsed.json --qpp outputs/phase1/D_qpp.json \\
                       --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json \\
                       --generated-pptx outputs/phase4/audittailor_generated.pptx --verify

It writes outputs/phase5/verification.json (measurement only; no repair).

Phase 6 (bounded, deterministic, extractive repair loop) is switched on with --repair (it runs Phase 5 first):

    python src/main.py --paper P.pdf --deck D.pptx --plan --n-slides 10 --draft --pptx --repair     # all six phases
    python src/main.py --parsed outputs/phase1/P_parsed.json --qpp outputs/phase1/D_qpp.json \\
                       --plan-json outputs/phase2/plan.json --draft-json outputs/phase3/draft.json \\
                       --generated-pptx outputs/phase4/audittailor_generated.pptx --repair

It writes outputs/phase6/final.pptx, final_verification.json, repair_history.json (+ final_draft.json, candidates/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser import ParsedPaper, parse_pdf          # noqa: E402
from planner import Plan, plan_slides, plan_from_phase1, build_story_plan  # noqa: E402
from qpp import QPP, extract_qpp                    # noqa: E402
from section_tagger import CANONICAL_SECTIONS       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs" / "phase1"
DEFAULT_PLAN_OUT = ROOT / "outputs" / "phase2" / "plan.json"
DEFAULT_DRAFT_OUT = ROOT / "outputs" / "phase3" / "draft.json"
DEFAULT_PPTX_OUT = ROOT / "outputs" / "phase4" / "audittailor_generated.pptx"
DEFAULT_VERIFY_OUT = ROOT / "outputs" / "phase5" / "verification.json"
DEFAULT_REPAIR_OUT = ROOT / "outputs" / "phase6"
DEFAULT_TEMPLATE = ROOT / "data" / "reference_decks" / "synthetic_reference.pptx"
DEFAULT_LAMBDA = 0.6     # demonstration default only (the proposal tunes lambda on a development split)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _trunc(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def report_paper(p: ParsedPaper, out_file: Path) -> None:
    wc = p.label_word_counts()
    total_w = sum(wc.values()) or 1
    print("=" * 78)
    print("[1] PAPER PARSE")
    print("=" * 78)
    print(f"File           : {p.path}")
    print(f"Title          : {_trunc(p.title, 100)}")
    print(f"Pages          : {p.n_pages}    Body font: {p.body_font_size:.1f} pt")
    print(f"Sections       : {len([s for s in p.sections if not s.excluded])} body  "
          f"({len([s for s in p.sections if s.excluded])} excluded)")
    print(f"Sentences      : {len(p.sentences)}  (dropped fragments: {p.dropped_fragments}, "
          f"margin lines removed: {p.removed_margin_lines}, captions removed: {len(p.captions)})")
    print(f"Abstract       : {'yes (' + str(len(p.abstract.split())) + ' words)' if p.abstract else 'not found'}")
    print()
    print(f"{'#':>2} {'num':<5} {'heading':<44} {'label':<11} {'src':<18} {'sent':>4} {'words':>5}")
    d = p.to_dict()["sections"]
    for s in d:
        lab = s["label"] or f"(excl: {s['excluded']})"
        print(f"{s['index']:>2} {str(s['number'] or ''):<5} {_trunc(s['heading'], 44):<44} "
              f"{lab:<11} {str(s['label_source'] or ''):<18} {s['n_sentences']:>4} {s['n_words']:>5}")
    print()
    print("Section length l_k by canonical label (P = labels with l_k > 0):")
    for k in CANONICAL_SECTIONS:
        share = wc[k] / total_w
        print(f"  {k:<11} {wc[k]:>6} words  {share:6.1%}  {'#' * int(round(share * 40))}")
    print(f"  P = {p.present_labels()}")
    print()
    print("First 3 sentences with pointers (page, section, id):")
    for s in p.sentences[:3]:
        print(f"  ({s.page}, {_trunc(s.section, 28)!r}, {s.id}) {_trunc(s.text, 80)}")
    print(f"\nJSON written: {out_file}\n")


def report_qpp(q: QPP, out_file: Path) -> None:
    print("=" * 78)
    print("[2] QUANTIFIED PREFERENCE PROFILE  theta = (w, pi, phi)")
    print("=" * 78)
    print(f"File           : {q.source}")
    print(f"Slides         : {q.n_slides} total, {q.n_content} content, "
          f"{q.n_slides - q.n_content} excluded")
    print(f"Fit estimator  : {q.fit_estimator}")
    print()
    print("par = body paragraphs (prose or bullet), bul = real bullet paragraphs, fig = picture/table")
    print(f"{'#':>2} {'title':<34} {'kind':<16} {'label':<11} {'src':<9} {'par':>3} {'bul':>3} {'fig':>3} {'eq':>2} {'fill':>5}")
    for s in q.slides:
        d = s.to_dict()
        fill = "-" if d["mean_fill"] is None else f"{d['mean_fill']:.2f}"
        print(f"{s.index:>2} {_trunc(s.title, 34):<34} {s.kind:<16} {str(s.label or '-'):<11} "
              f"{str(s.label_source or '-'):<9} {s.n_body_paragraphs:>3} {s.n_bullet_paragraphs:>3} {'Y' if s.has_figure else '.':>3} "
              f"{'Y' if s.has_equation else '.':>2} {fill:>5}")
    print()
    print("w  (section slide-share distribution):")
    for k in CANONICAL_SECTIONS:
        print(f"  {k:<11} c={q.counts[k]:<3} w={q.w[k]:.4f}  {'#' * int(round(q.w[k] * 40))}")
    print(f"  sum(w) = {sum(q.w.values()):.6f}")
    print(f"\npi (flow)      : {' -> '.join(q.pi) if q.pi else '(empty)'}")
    print("phi (style)    :")
    print(f"  eta     (fill ratio)                  = {q.eta:.4f}")
    print(f"  beta    (slides with >=3 paragraphs)  = {q.beta:.4f}")
    print(f"  rho     (slides with picture/table)   = {q.rho:.4f}   [charts not counted]")
    print(f"  epsilon (slides with equation)        = {q.epsilon:.4f}")
    print(f"mean bullets per content slide (b_bar)  = {q.mean_bullets:.4f}   "
          f"({q.total_bullets} bullet paragraphs / {q.n_content} content slides)")
    print(f"  (diagnostic) mean body paragraphs     = {q.mean_paragraphs:.4f}   [prose + bullets; basis of beta, NOT b_bar]")
    if q.warnings:
        print("\nWarnings:")
        for w in q.warnings:
            print(f"  - {w}")
    print(f"\nJSON written: {out_file}\n")


def report_plan(plan: Plan, paper_name: str, qpp_name: str, out_file: Path) -> None:
    bar = "=" * 40
    print(bar)
    print("AuditTailor - Phase 2")
    print("Budgeted Slide Planner")
    print(bar)
    print()
    print(f"Target paper: {paper_name}")
    print(f"Reference QPP: {qpp_name}")
    print(f"Total slides: {plan.total_slides}")
    print(f"Lambda: {float(plan.lam):.2f}")

    def block(title, values, fmt="{:.4f}"):
        print()
        print(title)
        print("-" * len(title))
        for k in CANONICAL_SECTIONS:
            tag = "   (absent from paper)" if k in plan.absent else ""
            print(f"{k:<13} : {fmt.format(values[k])}{tag}")

    d = plan.to_dict()
    block("PAPER DISTRIBUTION", d["paper_distribution"])
    block("REFERENCE DISTRIBUTION", d["reference_distribution"])
    block("REFERENCE DISTRIBUTION (restricted to paper, renormalised)", d["reference_distribution_restricted"])
    block("BLENDED DISTRIBUTION", d["blended_distribution"])
    print()
    print("LARGEST-REMAINDER ROUNDING")
    print("--------------------------")
    print(f"{'section':<13} {'N*q':>8} {'floor':>6} {'remainder':>10} {'extra':>6} {'slides':>7}")
    for k in plan.present:
        dd = d["allocation_detail"][k]
        print(f"{k:<13} {dd['raw_N_times_q']:>8.4f} {dd['floor']:>6} {dd['remainder']:>10.4f} "
              f"{'+1' if dd['extra_slide'] else '':>6} {dd['slides']:>7}")
    print()
    print("FINAL SLIDE ALLOCATION")
    print("----------------------")
    for k in CANONICAL_SECTIONS:
        tag = "   (absent from paper)" if k in plan.absent else ""
        print(f"{k:<13} : {plan.allocation[k]}{tag}")
    print()
    print(f"{'Total':<13} : {sum(plan.allocation.values())}")
    print(f"Section order : {' -> '.join(plan.order)}")
    if plan.inserted:
        print(f"Inserted (not in reference order): {', '.join(plan.inserted)}")
    for note in plan.fallbacks + plan.warnings:
        print(f"Note: {note}")
    print(f"\nJSON written: {out_file}\n")


def report_draft(draft: dict, out_file: Path) -> None:
    bar = "=" * 40
    p = draft["parameters"]
    print(bar)
    print("AuditTailor - Phase 3")
    print("Key-Free Extractive Drafting")
    print(bar)
    print()
    print(f"Slides planned: {draft['n_slides']}")
    print(f"Lambda_MMR: {p['lambda_mmr']:.2f}")
    ref = p["reference_mean_bullets"]
    print(f"Mean bullets/slide: {ref:.2f}" if isinstance(ref, (int, float)) else "Mean bullets/slide: n/a (reference deck has none)")
    print(f"Bullets per slide used (b_bar, rounded): {p['bullets_per_slide_b_bar']}")
    print(f"Candidate sentences: {draft['candidate_filter']['kept']} of {draft['candidate_filter']['parsed_sentences']} parsed")
    for s in draft["slides"]:
        print()
        print(f"SLIDE {s['slide_number']}")
        print("-" * len(f"SLIDE {s['slide_number']}"))
        print(f"Section: {s['section']}")
        if not s["bullets"]:
            print("(no bullets - not enough suitable sentences)")
        for i, b in enumerate(s["bullets"], 1):
            src = b["source"]
            print(f"{i}. {b['text']}")
            print(f"   Source: page {src['page']}, {src['sentence_id']}")
    print()
    print("SECTION BUDGET")
    print("--------------")
    print(f"{'section':<13} {'slides':>6} {'requested':>9} {'candidates':>10} {'selected':>8}")
    for k, v in draft["section_summary"].items():
        print(f"{k:<13} {v['slides']:>6} {v['requested_bullets']:>9} {v['candidates']:>10} {v['selected']:>8}")
    for line in draft["warnings"] + draft["notes"][1:]:
        print(line)
    print(f"\nDraft written to:\n{_rel(out_file)}\n")


def report_pptx(meta: dict, meta_file: Path) -> None:
    bar = "=" * 40
    print(bar)
    print("AuditTailor - Phase 4")
    print("Editable PPTX Generation")
    print(bar)
    print()
    print(f"Template: {_rel(Path(meta['inputs']['template']))}")
    print(f"Slides generated: {meta['n_slides']}  ({meta['n_content_slides']} content of "
          f"{meta['n_content_slides_planned']} planned)")
    print(f"Bullets: {meta['n_bullets']}   Source pointers retained: {meta['n_source_pointers']}"
          f"   (every bullet has one: {meta['all_bullets_have_source_pointer']})")
    print()
    for m in meta["slides"]:
        extra = f"{m['n_bullets']} bullet(s)" if m["kind"] == "content" else m["kind"]
        print(f"  slide {m['pptx_slide']:>2}  {m['title']:<26} {extra}")
    print()
    print(f"Overflow adjustments: {meta['n_overflow_adjustments']}   unresolved: {meta['n_unresolved_overflow']}")
    for o in meta["overflow_adjustments"]:
        print(f"  slide {o['slide']}: {o['font_size_start_pt']:.0f} -> {o['font_size_final_pt']:.0f} pt, "
              f"fill {o['fill_before']} -> {o['fill_after']}")
    for w in meta["warnings"] + meta["errors"]:
        print(f"Note: {w}")
    print(f"\nPPTX written to:\n{_rel(Path(meta['output_pptx']))}\nMetadata written to:\n{_rel(meta_file)}\n")


def report_verification(v: dict, out_file: Path) -> None:
    bar = "=" * 40
    print(bar)
    print("AuditTailor - Phase 5")
    print("Verification and Evaluation")
    print(bar)
    print()
    print(f"Generated deck: {_rel(Path(v['inputs']['pptx']))}")
    print()
    for label, key in (("Dalloc (allocation distance)", "dalloc"), ("Dorder (order distance)", "dorder"),
                       ("Dstyle (QPP style distance)", "dstyle"), ("Dpref", "dpref"), ("PF = 1 - Dpref", "pf"),
                       ("SR (lexical source support)", "sr"), ("NF (numeric fidelity)", "nf"),
                       ("O (overflow rate)", "overflow_rate"), ("J(S) (lower is better)", "objective")):
        print(f"{label:<32}: {v[key]:.4f}")
    d = v["details"]
    print()
    print(f"Source support : {d['sr']['supported_bullets']}/{d['sr']['total_bullets']} bullets (threshold {d['sr']['threshold']})")
    print(f"Numeric claims : {d['nf']['numeric_claims_correct']}/{d['nf']['numeric_claims_checked']} correct")
    print(f"Overflow       : {d['overflow']['overflow_count']}/{d['overflow']['total_checked']} text boxes")
    for w in v["warnings"]:
        print(f"Note: {w}")
    print(f"\nVerification written to:\n{_rel(out_file)}\n")


def report_repair(res: dict, out_dir: Path) -> None:
    bar = "=" * 40
    h, v = res["history"], res["verification"]
    print(bar)
    print("AuditTailor - Phase 6")
    print("Bounded Repair Loop")
    print(bar)
    print()
    print(f"Repair rounds used : {h['rounds_used']} (max {h['max_repair_rounds']})   stop: {h['stop_reason']}")
    print(f"Initial J(S)       : {h['initial']['J']:.4f}   failures: {h['initial']['failures'] or 'none'}")
    for r in h["rounds"]:
        types = ", ".join(a["repair_type"] + ("" if a["changed"] else " (no change)") for a in r["actions"])
        new_j = "n/a" if r["new_J"] is None else f"{r['new_J']:.4f}"
        print(f"  round {r['round']}: {types}   J {r['previous_J']:.4f} -> {new_j}   "
              f"improved={r['improved']} best={r['became_best']}")
    f = h["final"]
    print(f"Final J(S)         : {f['J']:.4f}   improved: {f['improved_over_initial']}   "
          f"selected: {f['selected_candidate']}")
    print(f"Final Dpref {v['dpref']:.4f}  PF {v['pf']:.4f}  SR {v['sr']:.4f}  NF {v['nf']:.4f}  O {v['overflow_rate']:.4f}")
    if f["remaining_failures"]:
        print(f"Criteria still failing: {', '.join(f['remaining_failures'])}")
    print(f"\nPhase 6 outputs written to:\n{_rel(out_dir)}\n")


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AuditTailor Phase 1: PDF parsing + QPP extraction (deterministic, key-free)")
    ap.add_argument("--paper", type=Path, help="target paper PDF")
    ap.add_argument("--deck", type=Path, help="reference deck PPTX")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Phase 1 output directory (default: {DEFAULT_OUT})")
    # ---- Phase 2 ----
    ap.add_argument("--plan", action="store_true", help="run the Phase 2 budgeted planner")
    ap.add_argument("--n-slides", type=int, help="N: desired number of content slides (required with --plan)")
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA,
                    help=f"blend weight in [0,1] (default {DEFAULT_LAMBDA}; 0 = paper structure, 1 = presenter)")
    ap.add_argument("--parsed", type=Path, help="Phase 1 parsed-paper JSON (alternative to --paper when planning)")
    ap.add_argument("--qpp", type=Path, help="Phase 1 QPP JSON (alternative to --deck when planning)")
    ap.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_OUT, help=f"plan JSON path (default: {DEFAULT_PLAN_OUT})")
    # ---- Phase 3 ----
    ap.add_argument("--draft", action="store_true", help="run Phase 3 key-free extractive drafting (TF-IDF + MMR)")
    ap.add_argument("--plan-json", type=Path, help="Phase 2 plan JSON (use with --draft instead of --plan)")
    ap.add_argument("--lambda-mmr", dest="lambda_mmr", type=float, default=0.70,
                    help="MMR relevance/diversity trade-off in [0,1] (default 0.70; implementation parameter, not learned)")
    ap.add_argument("--rel-weights", type=float, nargs=4, metavar=("TOPICAL", "KEYWORD", "SECTION", "EVIDENCE"),
                    default=(0.25, 0.25, 0.25, 0.25), help="relevance signal weights (normalised to sum 1; default equal)")
    ap.add_argument("--max-bullet-words", type=int, default=35, help="clause-cut sentences longer than this (default 35)")
    ap.add_argument("--no-handover", action="store_true", help="do not pass unmet bullet quota to the next-highest-q_k section")
    ap.add_argument("--draft-out", type=Path, default=DEFAULT_DRAFT_OUT, help=f"draft JSON path (default: {DEFAULT_DRAFT_OUT})")
    # ---- Phase 4 ----
    ap.add_argument("--pptx", action="store_true", help="run Phase 4: generate an editable PPTX from the plan + draft")
    ap.add_argument("--draft-json", type=Path, help="Phase 3 draft JSON (use with --pptx instead of --draft)")
    ap.add_argument("--template", type=Path, help=f"template PPTX (default: --deck if given, else {DEFAULT_TEMPLATE})")
    ap.add_argument("--pptx-out", type=Path, default=DEFAULT_PPTX_OUT, help=f"generated PPTX path (default: {DEFAULT_PPTX_OUT})")
    ap.add_argument("--min-font-size", type=float, default=None, help="overflow floor in pt (default: pptx_generator.MIN_FONT_SIZE = 16)")
    ap.add_argument("--no-source-slide", action="store_true", help="do not append the Source Index slide")
    # ---- Phase 5 ----
    ap.add_argument("--verify", action="store_true", help="run Phase 5 verification on the generated PPTX")
    ap.add_argument("--generated-pptx", type=Path, help="PPTX to verify (default: the --pptx output of this run)")
    ap.add_argument("--verify-out", type=Path, default=DEFAULT_VERIFY_OUT, help=f"verification JSON path (default: {DEFAULT_VERIFY_OUT})")
    ap.add_argument("--dpref-weights", type=float, nargs=3, metavar=("ALLOC", "ORDER", "STYLE"), default=None,
                    help="lambda_1..3 of Dpref, must sum to 1 (default 0.4 0.2 0.4)")
    ap.add_argument("--objective-weights", type=float, nargs=3, metavar=("MU", "NU", "XI"), default=None,
                    help="mu, nu, xi of J(S) (default 0.5 1 1)")
    ap.add_argument("--sr-threshold", type=float, default=None, help="lexical cosine threshold for SR (default 0.70)")
    # ---- Phase 6 ----
    ap.add_argument("--repair", action="store_true", help="run Phase 6: bounded repair loop (implies Phase 5 verification)")
    ap.add_argument("--repair-out", type=Path, default=DEFAULT_REPAIR_OUT, help=f"Phase 6 output directory (default: {DEFAULT_REPAIR_OUT})")
    ap.add_argument("--max-repair-rounds", type=int, default=None, help="repair rounds bound (default repair.MAX_REPAIR_ROUNDS = 3)")
    ap.add_argument("--max-overflow-rate", type=float, default=None, help="repair target for O (default 0.0)")
    ap.add_argument("--min-support", type=float, default=None, help="repair target for SR (default 1.0)")
    ap.add_argument("--min-nf", type=float, default=None, help="repair target for NF (default 1.0)")
    ap.add_argument("--max-pref-distance", type=float, default=None, help="repair target for Dpref (default repair.MAX_PREFERENCE_DISTANCE)")
    args = ap.parse_args(argv)

    phase4_only = args.pptx and args.draft_json and not (args.paper or args.parsed)
    if not (args.paper or args.deck or args.parsed or args.qpp or phase4_only):
        ap.error("give at least one of --paper / --deck / --parsed / --qpp")
    if args.plan and args.n_slides is None:
        ap.error("--plan requires --n-slides N")
    if args.n_slides is not None and not args.plan:
        ap.error("--n-slides is only used with --plan")
    if args.plan_json and not (args.draft or args.pptx or args.verify or args.repair):
        ap.error("--plan-json is only used with --draft, --pptx, --verify or --repair")
    if args.draft_json and not (args.pptx or args.verify or args.repair):
        ap.error("--draft-json is only used with --pptx, --verify or --repair")
    if args.draft_json and args.draft:
        ap.error("--draft-json (use an existing draft) and --draft (create one) are mutually exclusive")
    if args.pptx and not (args.draft or args.draft_json):
        ap.error("--pptx needs a Phase 3 draft: add --draft (run Phase 3) or --draft-json draft.json")

    flag = "--repair" if args.repair and not args.verify else "--verify"
    if args.repair:
        args.verify = True                                 # Phase 6 starts from a Phase 5 verification
    if args.verify and not (args.pptx or args.generated_pptx):
        ap.error(f"{flag} needs a generated deck: add --pptx (run Phase 4) or --generated-pptx deck.pptx")
    if args.verify and not (args.draft or args.draft_json):
        ap.error(f"{flag} needs a Phase 3 draft: add --draft or --draft-json draft.json")
    if args.verify and args.generated_pptx and args.pptx:
        ap.error("--generated-pptx and --pptx are mutually exclusive")

    paper = qpp = plan = draft = None
    try:
        if args.paper:
            paper = parse_pdf(args.paper)
            out = args.out / f"{args.paper.stem}_parsed.json"
            _write_json(paper.to_dict(), out)
            report_paper(paper, out)
        if args.deck:
            qpp = extract_qpp(args.deck)
            out = args.out / f"{args.deck.stem}_qpp.json"
            _write_json(qpp.to_dict(), out)
            report_qpp(qpp, out)

        if args.plan:
            # ---- inputs: this run's Phase 1 objects, or Phase 1 JSON files ----
            if paper is not None:
                words, paper_name = paper.label_word_counts(), str(args.paper)
            elif args.parsed:
                words, paper_name = _load_json(args.parsed)["label_word_counts_l"], str(args.parsed)
            else:
                raise ValueError("--plan needs the target paper: give --paper (PDF) or --parsed (Phase 1 JSON)")
            if qpp is not None:
                w, pi, qpp_name = qpp.w, qpp.pi, str(args.deck)
            elif args.qpp:
                theta = _load_json(args.qpp)["theta"]
                w, pi, qpp_name = theta["w"], theta["pi"], str(args.qpp)
            else:
                raise ValueError("--plan needs the reference profile: give --deck (PPTX) or --qpp (Phase 1 JSON)")
            plan = plan_slides(words, w, pi, args.n_slides, args.lam)
            parsed_for_story = None
            if paper is not None:
                parsed_for_story = paper.to_dict()
            elif args.parsed:
                parsed_for_story = _load_json(args.parsed)
            if parsed_for_story is not None:
                try:
                    plan.story = build_story_plan(
                        parsed_for_story, plan.allocation, plan.presented_order
                    )
                except Exception as e:
                    plan.warnings.append(f"story plan skipped: {e}")
            _write_json(plan.to_dict(), args.plan_out)
            report_plan(plan, paper_name, qpp_name, args.plan_out)

        if args.draft:
            from extractor import draft_deck        # lazy: Phases 1-2 do not need scikit-learn
            if paper is not None:
                parsed_d, parsed_name = paper.to_dict(), str(args.paper)
            elif args.parsed:
                parsed_d, parsed_name = _load_json(args.parsed), str(args.parsed)
            else:
                raise ValueError("--draft needs the target paper: give --parsed (Phase 1 JSON) or --paper (PDF)")
            if qpp is not None:
                qpp_d, qpp_name = qpp.to_dict(), str(args.deck)
            elif args.qpp:
                qpp_d, qpp_name = _load_json(args.qpp), str(args.qpp)
            else:
                raise ValueError("--draft needs the reference profile: give --qpp (Phase 1 JSON) or --deck (PPTX)")
            if plan is not None:
                plan_d, plan_name = plan.to_dict(), "(planned in this run)"
            elif args.plan_json:
                plan_d, plan_name = _load_json(args.plan_json), str(args.plan_json)
            else:
                raise ValueError("--draft needs a Phase 2 plan: give --plan-json (plan.json) or --plan --n-slides N")
            draft = draft_deck(parsed_d, qpp_d, plan_d, lambda_m=args.lambda_mmr, rel_weights=args.rel_weights,
                               max_bullet_words=args.max_bullet_words, handover=not args.no_handover,
                               inputs={"parsed": parsed_name, "qpp": qpp_name, "plan": plan_name})
            _write_json(draft, args.draft_out)
            report_draft(draft, args.draft_out)

        if args.pptx:
            from pptx_generator import generate_pptx      # lazy: Phases 1-3 do not load the generator
            if args.draft_json:
                draft, draft_name = _load_json(args.draft_json), str(args.draft_json)
            else:
                draft, draft_name = draft, "(drafted in this run)"
            if plan is not None:
                plan_d, plan_name = plan.to_dict(), "(planned in this run)"
            elif args.plan_json:
                plan_d, plan_name = _load_json(args.plan_json), str(args.plan_json)
            else:
                raise ValueError("--pptx needs a Phase 2 plan: give --plan-json (plan.json) or --plan --n-slides N")
            template = args.template or args.deck or DEFAULT_TEMPLATE
            kw = {} if args.min_font_size is None else {"min_font_size": args.min_font_size}
            meta = generate_pptx(draft, plan_d, template, args.pptx_out, source_slide=not args.no_source_slide,
                                 inputs={"plan": plan_name, "draft": draft_name}, **kw)
            meta_file = args.pptx_out.parent / "pptx_metadata.json"
            _write_json(meta, meta_file)
            report_pptx(meta, meta_file)

        if args.verify:
            from verifier import verify                   # lazy: earlier phases do not load the verifier
            if paper is not None:
                parsed_d, parsed_name = paper.to_dict(), str(args.paper)
            elif args.parsed:
                parsed_d, parsed_name = _load_json(args.parsed), str(args.parsed)
            else:
                raise ValueError("--verify needs the target paper: give --paper (PDF) or --parsed (Phase 1 JSON)")
            if qpp is not None:
                qpp_d, qpp_name = qpp.to_dict(), str(args.deck)
            elif args.qpp:
                qpp_d, qpp_name = _load_json(args.qpp), str(args.qpp)
            else:
                raise ValueError("--verify needs the reference profile: give --deck (PPTX) or --qpp (Phase 1 JSON)")
            if plan is not None:
                plan_d, plan_name = plan.to_dict(), "(planned in this run)"
            elif args.plan_json:
                plan_d, plan_name = _load_json(args.plan_json), str(args.plan_json)
            else:
                raise ValueError("--verify needs a Phase 2 plan: give --plan-json (plan.json) or --plan --n-slides N")
            if args.draft_json:
                draft_v, draft_name = _load_json(args.draft_json), str(args.draft_json)
            else:
                draft_v, draft_name = draft, "(drafted in this run)"
            deck = args.generated_pptx or args.pptx_out
            kw = {}
            if args.dpref_weights:
                kw["pref_weights"] = dict(zip(("alloc", "order", "style"), args.dpref_weights))
            if args.objective_weights:
                kw["objective_weights"] = dict(zip(("mu", "nu", "xi"), args.objective_weights))
            if args.sr_threshold is not None:
                kw["sr_threshold"] = args.sr_threshold
            if args.min_font_size is not None:
                kw["min_font_size"] = args.min_font_size
            result = verify(parsed_d, qpp_d, plan_d, draft_v, deck, inputs={
                "parsed": parsed_name, "reference_qpp": qpp_name, "plan": plan_name, "draft": draft_name}, **kw)
            _write_json(result, args.verify_out)
            report_verification(result, args.verify_out)

            if args.repair:
                import repair as R                       # lazy: earlier phases do not load the repair module
                cfg_kw = {k: v for k, v in (("max_rounds", args.max_repair_rounds), ("max_overflow_rate", args.max_overflow_rate),
                                            ("min_support_ratio", args.min_support), ("min_numeric_fidelity", args.min_nf),
                                            ("max_preference_distance", args.max_pref_distance)) if v is not None}
                res = R.repair_loop(parsed_d, qpp_d, plan_d, draft_v, deck, args.repair_out,
                                    template=args.template or args.deck or DEFAULT_TEMPLATE,
                                    config=R.RepairConfig(**cfg_kw), verify_kw=kw,
                                    inputs={"parsed": parsed_name, "reference_qpp": qpp_name, "plan": plan_name,
                                            "draft": draft_name, "pptx": deck.name})
                report_repair(res, args.repair_out)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    sys.exit(main())

"""
pipeline.py -- AuditTailor orchestration layer (no methodology of its own).

    run_pipeline(paper_pdf, reference_pptx, output_dir, config)  ->  PipelineResult

Chains the existing Phase 1-6 modules exactly as ``main.py`` does (parse_pdf, extract_qpp, plan_slides, draft_deck,
generate_pptx, verify, repair_loop), using whatever file paths it is given.  The web interface (app.py) only calls
this module, so the UI can be replaced without touching the research pipeline.  Key-free and deterministic: no LLM,
no API, no network.

Output layout (same as the CLI): <output_dir>/phase1 ... phase6, with the deliverable at phase6/final.pptx.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
import sys                                                      # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

import builtin_styles as BS                                     # noqa: E402
import repair as R                                              # noqa: E402
from extractor import draft_deck                                # noqa: E402
from main import DEFAULT_LAMBDA                                 # noqa: E402
from parser import parse_pdf                                    # noqa: E402
from planner import plan_slides, build_story_plan               # noqa: E402
from pptx_generator import generate_pptx                        # noqa: E402
from qpp import extract_qpp                                     # noqa: E402
from verifier import verify                                     # noqa: E402

try:                                                            # PyMuPDF is imported as pymupdf (new) or fitz (old)
    import pymupdf as _pymupdf                                  # noqa: E402
except ImportError:                                             # pragma: no cover
    import fitz as _pymupdf                                     # noqa: E402

DEFAULT_N_SLIDES = 10                 # same demonstration value as the CLI examples / tests (CLI requires --n-slides)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # sanity limit for uploaded files

# progress stages, in order: (key, label shown to the user once the stage has finished)
STAGES = [("parse", "Paper parsed"), ("preferences", "Preferences extracted"), ("plan", "Slide plan created"),
          ("draft", "Source-grounded content drafted"), ("pptx", "Presentation generated"),
          ("verify", "Presentation verified"), ("repair", "Repair completed")]


class PipelineError(Exception):
    """Friendly, stage-aware failure (the original exception is chained as __cause__)."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


_STAGE_USER_TEXT = {
    "input": "An input file could not be found.",
    "parse": "We could not read the research paper.",
    "preferences": "We could not read the reference presentation.",
    "plan": "We could not plan the slides.",
    "draft": "We could not draft the slide content.",
    "pptx": "We could not build the presentation.",
    "verify": "We could not verify the presentation.",
    "repair": "We could not finish the repair step.",
}


def describe_error(e: BaseException) -> str:
    """Plain-language message for the interface: never a traceback or an exception class name.
    Our own validation errors (ValueError) are already written for users and are shown as they are."""
    if isinstance(e, PipelineError):
        text = _STAGE_USER_TEXT.get(e.stage, "Something went wrong while generating the presentation.")
        cause = e.__cause__
        if isinstance(cause, (ValueError, FileNotFoundError)) and str(cause):
            text += f" {cause}"
        else:
            text += " Please check that the files are valid and try again."
        return text
    if isinstance(e, (ValueError, FileNotFoundError)):
        return str(e)
    return "Something went wrong while generating the presentation. Please check your files and try again."


@dataclass
class PipelineConfig:
    """User-tunable settings.  Defaults come from the existing modules (nothing new is invented here)."""
    n_slides: int = DEFAULT_N_SLIDES
    lam: float = DEFAULT_LAMBDA
    max_repair_rounds: int = R.MAX_REPAIR_ROUNDS
    min_support_ratio: float = R.MIN_SUPPORT_RATIO
    min_numeric_fidelity: float = R.MIN_NUMERIC_FIDELITY
    max_overflow_rate: float = R.MAX_OVERFLOW_RATE

    def repair_config(self) -> R.RepairConfig:
        return R.RepairConfig(max_rounds=self.max_repair_rounds, max_overflow_rate=self.max_overflow_rate,
                              min_support_ratio=self.min_support_ratio, min_numeric_fidelity=self.min_numeric_fidelity)


@dataclass(frozen=True)
class ReferenceInfo:
    """Which reference presentation taught the preferences (shown after generation, for reproducibility)."""
    kind: str                       # "builtin" | "custom"
    label: str                      # e.g. "Academic Research (built-in)" / "Custom uploaded presentation"
    style_id: Optional[str] = None  # built-in style id (None for a custom upload)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "style_id": self.style_id, "description": self.description}


@dataclass
class StagedInputs:
    work_dir: Path
    paper_pdf: Path
    reference_pptx: Path
    reference: ReferenceInfo


@dataclass
class PipelineResult:
    output_dir: Path
    final_pptx: Path
    verification_json: Path
    repair_history_json: Path
    verification: Dict[str, Any]
    history: Dict[str, Any]
    n_slides: int
    stages_done: List[str] = field(default_factory=list)
    reference: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- uploads (untrusted input) #
def stage_uploads(paper_bytes: bytes, paper_name: str, deck_bytes: bytes, deck_name: str,
                  base_dir: Optional[Path] = None) -> Path:
    """Validate uploaded bytes and write them to a NEW temporary directory under safe generated names
    (``paper.pdf`` / ``reference.pptx``); the user-supplied filenames are never used as paths.  Returns the directory.
    Files are only read as data by the parsers, never executed."""
    _check_upload("research paper", paper_name, paper_bytes, ".pdf")
    _check_upload("reference presentation", deck_name, deck_bytes, ".pptx")
    work = Path(tempfile.mkdtemp(prefix="audittailor_", dir=str(base_dir) if base_dir else None))
    (work / "paper.pdf").write_bytes(paper_bytes)
    (work / "reference.pptx").write_bytes(deck_bytes)
    return work


def _check_upload(label: str, name: str, data: bytes, ext: str) -> None:
    """Extension, size and file-signature checks for an uploaded file (shared by every staging function)."""
    if not str(name).lower().endswith(ext):
        raise ValueError(f"The {label} must be a {ext.upper()[1:]} file (got '{Path(str(name)).name}').")
    if not data:
        raise ValueError(f"The {label} is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The {label} is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    ok = data.lstrip()[:5] == b"%PDF-" if ext == ".pdf" else data[:4] == b"PK\x03\x04"
    if not ok:
        raise ValueError(f"The {label} does not look like a valid {ext.upper()[1:]} file.")
    if ext == ".pptx":
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as z:
                if "[Content_Types].xml" not in z.namelist() or not any(n.startswith("ppt/") for n in z.namelist()):
                    raise ValueError(f"The {label} is not a valid PPTX file.")
        except zipfile.BadZipFile as e:
            raise ValueError(f"The {label} is not a valid PPTX file.") from e


def validate_pdf_readable(path: Path) -> None:
    """The paper can be opened and has selectable text (no OCR: a scanned PDF cannot be used).  I/O check only."""
    try:
        doc = _pymupdf.open(str(path))
    except Exception:
        raise ValueError("The research paper could not be opened. The PDF may be damaged.") from None
    try:
        if getattr(doc, "needs_pass", False):
            raise ValueError("The research paper is password-protected. Please upload an unprotected PDF.")
        if doc.page_count < 1:
            raise ValueError("The research paper has no pages.")
        if not any(doc[i].get_text().strip() for i in range(min(doc.page_count, 6))):
            raise ValueError("No selectable text was found in the research paper. Scanned PDFs are not supported "
                             "(AuditTailor does not use OCR).")
    finally:
        doc.close()


def validate_reference_readable(path: Path, what: str = "reference presentation") -> None:
    """The reference deck can be opened and a preference profile (QPP) can be extracted from it."""
    try:
        n_content = extract_qpp(path).to_dict()["n_content_slides"]
    except Exception as e:
        raise ValueError(f"The {what} could not be analysed ({e}). Please use a normal PowerPoint (.pptx) file that "
                         f"contains at least one content slide.") from None
    if n_content < 1:                       # Phase 1 returns an all-zero profile (with a warning) for such a deck
        raise ValueError(f"The {what} could not be analysed: it contains no content slides to learn from (title, "
                         f"outline, references and closing slides do not count).")


def stage_inputs(paper_bytes: bytes, paper_name: str, *, style_id: str = BS.DEFAULT_STYLE_ID,
                 custom_bytes: Optional[bytes] = None, custom_name: Optional[str] = None,
                 base_dir: Optional[Path] = None, builtin_dir: Optional[Path] = None) -> StagedInputs:
    """Validate the inputs and stage them in a NEW temporary directory (``paper.pdf`` / ``reference.pptx``).

    The reference is the user's own deck when ``custom_bytes`` is given (it OVERRIDES the built-in style), otherwise
    the bundled deck of ``style_id`` (default: Academic Research).  The bundled files are only copied, never modified.
    Raises ValueError with a user-readable message; nothing is left behind on failure."""
    _check_upload("research paper", paper_name, paper_bytes, ".pdf")
    if custom_bytes is not None:
        _check_upload("reference presentation", custom_name or "", custom_bytes, ".pptx")
        ref_bytes = custom_bytes
        info = ReferenceInfo("custom", "Custom uploaded presentation", None,
                             "Preferences were extracted from the presentation uploaded by the user.")
    else:
        style = BS.get_style(style_id)
        src = BS.builtin_path(style_id, builtin_dir)
        if not src.is_file():
            raise ValueError(f"The built-in style '{style.label}' is not installed (missing file "
                             f"'{src.name}'). Please choose another style or upload your own reference presentation.")
        ref_bytes = src.read_bytes()
        info = ReferenceInfo("builtin", f"{style.label} (built-in)", style.id, style.description)
    work = Path(tempfile.mkdtemp(prefix="audittailor_", dir=str(base_dir) if base_dir else None))
    try:
        (work / "paper.pdf").write_bytes(paper_bytes)
        (work / "reference.pptx").write_bytes(ref_bytes)
        validate_pdf_readable(work / "paper.pdf")
        validate_reference_readable(work / "reference.pptx",
                                    "reference presentation" if info.kind == "custom" else "built-in reference presentation")
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return StagedInputs(work, work / "paper.pdf", work / "reference.pptx", info)


def cleanup(work_dir: Optional[Path]) -> None:
    """Remove a temporary working directory (best effort)."""
    if work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------------------------------------------------------- pipeline #
def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def display_name(name: str, default: str = "paper.pdf") -> str:
    """Sanitised file name for DISPLAY only (e.g. the title-slide subtitle); never used as a path."""
    base = re.sub(r"[^\w.\- ()]", "_", Path(str(name).replace("\\", "/")).name)[:80].strip()
    return base or default


def run_pipeline(paper_pdf: Path, reference_pptx: Path, output_dir: Path, config: Optional[PipelineConfig] = None,
                 progress: Optional[Callable[[str, str], None]] = None, paper_name: Optional[str] = None,
                 reference: Optional[ReferenceInfo] = None, template_pptx: Optional[Path] = None) -> PipelineResult:
    """Run Phases 1-6 on the given files.  ``progress(stage_key, label)`` is called after each stage completes.
    ``paper_name``: original file name of the paper, shown on the title slide (defaults to the file's own name).
    ``reference``: which reference was used (recorded in the Phase 5/6 provenance and returned in the result).
    ``template_pptx``: visual template for Phases 4/6; defaults to the reference deck itself (the project's existing
    behaviour), but the two roles are independent: the reference teaches preferences (QPP), the template gives the look."""
    cfg = config or PipelineConfig()
    ref_name = reference.label if reference else "reference.pptx"
    shown = display_name(paper_name or Path(paper_pdf).name)
    paper_pdf, reference_pptx, out = Path(paper_pdf), Path(reference_pptx), Path(output_dir)
    template = Path(template_pptx) if template_pptx else reference_pptx
    for p, what in ((paper_pdf, "research paper"), (reference_pptx, "reference presentation"), (template, "template")):
        if not p.is_file():
            raise PipelineError("input", f"The {what} could not be found.")
    done: List[str] = []
    labels = dict(STAGES)

    def tick(key: str) -> None:
        done.append(key)
        if progress:
            progress(key, labels[key])

    stage = "parse"
    try:
        paper = parse_pdf(paper_pdf)                                          # Phase 1: paper
        parsed = paper.to_dict()
        _dump(parsed, out / "phase1" / "paper_parsed.json")
        tick("parse")

        stage = "preferences"
        qpp = extract_qpp(reference_pptx).to_dict()                           # Phase 1: reference preferences
        _dump(qpp, out / "phase1" / "reference_qpp.json")
        tick("preferences")

        stage = "plan"
        plan_obj = plan_slides(paper.label_word_counts(), qpp["theta"]["w"], qpp["theta"]["pi"],
                               cfg.n_slides, cfg.lam)                         # Phase 2
        try:
            plan_obj.story = build_story_plan(
                parsed, plan_obj.allocation, plan_obj.presented_order
            )
        except Exception as e:
            plan_obj.warnings.append(f"story plan skipped: {e}")
        plan = plan_obj.to_dict()
        _dump(plan, out / "phase2" / "plan.json")
        tick("plan")

        stage = "draft"
        draft = draft_deck(parsed, qpp, plan, inputs={"parsed": shown, "qpp": ref_name,
                                                     "plan": "(planned in this run)"})   # Phase 3
        _dump(draft, out / "phase3" / "draft.json")
        tick("draft")

        stage = "pptx"
        deck = out / "phase4" / "audittailor_generated.pptx"
        meta = generate_pptx(draft, plan, template, deck, inputs={"plan": "(planned in this run)",
                                                                       "draft": "(drafted in this run)"})   # Phase 4
        _dump(meta, out / "phase4" / "pptx_metadata.json")
        tick("pptx")

        stage = "verify"
        ver = verify(parsed, qpp, plan, draft, deck, inputs={"parsed": shown, "reference_qpp": ref_name})  # 5
        _dump(ver, out / "phase5" / "verification.json")
        tick("verify")

        stage = "repair"
        res = R.repair_loop(parsed, qpp, plan, draft, deck, out / "phase6", template=template,
                            config=cfg.repair_config(),
                            inputs={"parsed": shown, "reference_qpp": ref_name})          # Phase 6
        tick("repair")
    except PipelineError:
        raise
    except Exception as e:                                                    # surfaced with the failing stage
        raise PipelineError(stage, _friendly(stage, e)) from e

    p6 = out / "phase6"
    from pptx import Presentation
    return PipelineResult(
        output_dir=out, final_pptx=p6 / "final.pptx", verification_json=p6 / "final_verification.json",
        repair_history_json=p6 / "repair_history.json", verification=res["verification"], history=res["history"],
        n_slides=len(Presentation(str(p6 / "final.pptx")).slides), stages_done=done,
        reference=reference.to_dict() if reference else None)


def _friendly(stage: str, e: Exception) -> str:
    what = {"parse": "reading the research paper", "preferences": "reading the reference presentation",
            "plan": "planning the slides", "draft": "drafting the slide content",
            "pptx": "building the presentation", "verify": "verifying the presentation",
            "repair": "repairing the presentation"}.get(stage, "processing")
    return f"Something went wrong while {what}: {type(e).__name__}: {e}"


# ------------------------------------------------------------------------------- display helpers #
_REPAIR_LABELS = {"overflow": "Overflow adjustment", "support_fidelity": "Source-support / numeric correction",
                  "preference": "Slide structure (section labels / order / allocation) correction"}


def summarize_repairs(history: Dict[str, Any]) -> Dict[str, Any]:
    """Facts for the 'Repair Details' section, taken only from repair_history.json (no invented content)."""
    final = history["final"]
    applied = [_REPAIR_LABELS.get(t, t) for t in final.get("repair_types_applied", [])]
    return {"rounds_used": history["rounds_used"], "applied": applied, "any_repairs": bool(applied),
            "stop_reason": history["stop_reason"], "initial_J": history["initial"]["J"], "final_J": final["J"],
            "improved": final["improved_over_initial"], "remaining_failures": final.get("remaining_failures", [])}


def headline_metrics(ver: Dict[str, Any], n_slides: int) -> Dict[str, Any]:
    """Values shown on the result screen, straight from final_verification.json."""
    return {"Slides generated": n_slides, "Source Support": ver["sr"], "Numeric Fidelity": ver["nf"],
            "Overflow Rate": ver["overflow_rate"], "Preference Fidelity": ver["pf"], "Final Objective Score": ver["objective"]}


# Every figure shown on the result screen: (label, key in final_verification.json)
DETAIL_METRICS = (("Dalloc", "dalloc"), ("Dorder", "dorder"), ("Dstyle", "dstyle"), ("Dpref", "dpref"), ("PF", "pf"),
                  ("SR", "sr"), ("NF", "nf"), ("O", "overflow_rate"), ("J(S)", "objective"))


def detailed_metrics(ver: Dict[str, Any]) -> Dict[str, float]:
    """Dalloc, Dorder, Dstyle, Dpref, PF, SR, NF, O, J(S) read straight from final_verification.json."""
    return {label: ver[key] for label, key in DETAIL_METRICS}


def repair_round_rows(history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per repair round, from repair_history.json (empty list when no repair round was needed)."""
    rows = []
    for rr in history.get("rounds", []):
        rows.append({"Round": rr["round"], "Repair types tried": ", ".join(rr.get("families_attempted", [])),
                     "Actions that changed the deck": sum(1 for a in rr.get("actions", []) if a.get("changed")),
                     "J before": rr.get("previous_J"), "J after": rr.get("new_J"),
                     "Kept (J decreased)": bool(rr.get("improved")), "Note": rr.get("note", "")})
    return rows

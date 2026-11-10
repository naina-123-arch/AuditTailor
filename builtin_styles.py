"""
builtin_styles.py -- registry of the bundled reference presentations ("presentation styles").

A built-in style is just a bundled reference PPTX (data/reference_decks/builtin/<file>) that Phase 1 reads exactly like
a user-supplied reference deck: the Quantified Preference Profile (section allocation, order, fill ratio, bullet /
paragraph density, picture/table and equation usage) is extracted from it, and the target paper supplies the content.
The styles are NOT scientifically validated; they are convenience profiles for users who have no suitable reference
presentation of their own.  Selecting a style never bypasses Phases 1-6: it only chooses which reference file is used.

The generator for the bundled files is tools/build_builtin_decks.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
BUILTIN_DIR = ROOT / "data" / "reference_decks" / "builtin"


@dataclass(frozen=True)
class BuiltinStyle:
    id: str
    label: str
    description: str
    filename: str


STYLES = (
    BuiltinStyle("academic_research", "Academic Research",
                 "Structured research presentation with balanced sections and concise academic bullets.",
                 "academic_research.pptx"),
    BuiltinStyle("technical_ai", "Technical / AI",
                 "Technical presentation with stronger emphasis on methods, architecture and results.",
                 "technical_ai.pptx"),
    BuiltinStyle("conference", "Conference",
                 "Compact presentation with concise content and stronger results emphasis.",
                 "conference.pptx"),
    BuiltinStyle("minimal_professional", "Minimal Professional",
                 "Fewer bullets, more whitespace and concise explanations.",
                 "minimal_professional.pptx"),
    BuiltinStyle("detailed_research", "Detailed Research",
                 "More content per section and more detailed explanations.",
                 "detailed_research.pptx"),
)
DEFAULT_STYLE_ID = "academic_research"
_BY_ID = {s.id: s for s in STYLES}


def get_style(style_id: str) -> BuiltinStyle:
    try:
        return _BY_ID[style_id]
    except KeyError:
        raise ValueError(f"Unknown presentation style '{style_id}'.") from None


def builtin_path(style_id: str, builtin_dir: Optional[Path] = None) -> Path:
    """Path of the bundled reference deck for a style (may not exist; see :func:`is_available`)."""
    return Path(builtin_dir or BUILTIN_DIR) / get_style(style_id).filename


def is_available(style_id: str, builtin_dir: Optional[Path] = None) -> bool:
    return builtin_path(style_id, builtin_dir).is_file()


def available_styles(builtin_dir: Optional[Path] = None) -> List[BuiltinStyle]:
    return [s for s in STYLES if is_available(s.id, builtin_dir)]

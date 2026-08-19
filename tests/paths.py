"""
Locating the shared data and fixtures.

This pack lives in two places: as a directory inside the development monorepo,
and as its own standalone repository (which is what the ComfyUI registry wants —
the pack at the repo root). The two layouts put `fixtures/` and `shared/` at
different depths, so the tests search upward for them rather than counting
parent directories, and the same test files work unmodified in both.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_upward(*, marker: str, limit: int = 4) -> Path:
    """The nearest ancestor containing `marker`, searching from the tests dir."""
    for parent in [HERE, *HERE.parents][: limit + 1]:
        candidate = parent / marker
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find {marker!r} above {HERE} — the repository layout has changed"
    )


def fixtures_dir() -> Path:
    """The frozen parity fixtures (core-spec §7)."""
    return _find_upward(marker="fixtures/expected").parent


def shared_dir() -> Path:
    """The single source of truth for palette, charset and subsets."""
    return _find_upward(marker="shared/palette.json").parent


def core_spec() -> Path:
    """The normative algorithm document."""
    return _find_upward(marker="core-spec.md")

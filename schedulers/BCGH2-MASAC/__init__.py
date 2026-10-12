"""Canonical BCGH2-MASAC import path, backed by the historical MASCA directory."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "BCGH2-MASCA")]

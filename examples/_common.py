"""Shared helpers for the runnable atlas case studies (headless plotting)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Point matplotlib at a writable cache dir before first import (avoids the
# "config dir not writable" warning when HOME/.config is read-only).
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


def output_dir() -> Path:
    """The ``output/`` directory at the repository root (created if missing)."""
    out = Path(__file__).resolve().parent.parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig(fig, name):
    """Save ``fig`` to ``output/<name>``, close it, and return the path."""
    path = output_dir() / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

"""Shared helpers for running rPPG-Toolbox code from this repo.

These scripts run in the TOOLBOX environment (Python 3.8, see README), not in
our own one. They do not modify the toolbox; they work around two things:

1. ``main.py`` imports every trainer, including PhysMamba, which needs the
   ``mamba-ssm`` package. mamba-ssm only builds on Linux + CUDA, so on Windows
   (or without it) the toolbox would not start at all. If it is missing a
   stand-in module (stubs/mamba_ssm) is used, which only makes PhysMamba
   unusable - none of the models in our plan.
2. The toolbox loads its Haar-cascade face detector from a path relative to
   the working directory, so it must run from inside ``external/rPPG-Toolbox``.
   Our configs use paths relative to the repo root (``data/...``,
   ``cache/...``), so they are rewritten to absolute paths first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLBOX = REPO / "external" / "rPPG-Toolbox"
RESOLVED_CONFIG_DIR = REPO / "cache" / "toolbox" / "configs"

# Config keys whose values are filesystem paths.
PATH_KEYS = {"DATA_PATH", "CACHED_PATH", "FILE_LIST_PATH", "PATH", "MODEL_PATH", "MODEL_DIR", "OUTPUT_SAVE_DIR"}


def install_mamba_stub() -> None:
    """Fall back to stubs/mamba_ssm when the real package is not installed.

    The stub lives on disk and is added to the END of sys.path, so a real
    installation always wins, and worker processes (which inherit sys.path
    but not in-memory modules) find it too.
    """
    stubs = str(Path(__file__).resolve().parent / "stubs")
    if stubs not in sys.path:
        sys.path.append(stubs)


def resolve_config(config_file: str) -> str:
    """Copy ``config_file`` with every relative path made absolute (relative to the repo root)."""
    import yaml

    src = Path(config_file).resolve()
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in PATH_KEYS and isinstance(v, str) and v and not os.path.isabs(v):
                    node[k] = str((REPO / v).resolve())
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    RESOLVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = RESOLVED_CONFIG_DIR / src.name
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(out)


def enter_toolbox() -> None:
    """Make toolbox modules importable and switch to its directory."""
    os.environ.setdefault("MPLBACKEND", "Agg")  # the toolbox only saves figures; never open GUI windows
    install_mamba_stub()
    # Worker processes started with "spawn" (always the case on Windows) re-import the
    # main script by path, so it must stay valid after the chdir below.
    main = sys.modules["__main__"]
    if getattr(main, "__file__", None):
        main.__file__ = os.path.abspath(main.__file__)
    sys.argv[0] = os.path.abspath(sys.argv[0])
    sys.path.insert(0, str(TOOLBOX))
    os.chdir(TOOLBOX)

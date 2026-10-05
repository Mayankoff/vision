"""Run rPPG-Toolbox's main.py with one of our configs (toolbox environment).

    python scripts/toolbox/run_toolbox.py --config_file configs/toolbox/UBFC-rPPG_UNSUPERVISED.yaml

Same as ``python main.py --config_file ...`` inside the toolbox, but works on
Windows / without mamba-ssm and accepts configs with repo-relative paths (see
toolbox_env.py). Use it for training and testing the deep models (Phase 3).
For classical methods use dump_unsupervised.py, which also saves per-video
predictions.
"""

import argparse
import runpy
import sys

from toolbox_env import TOOLBOX, enter_toolbox, resolve_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    args, rest = ap.parse_known_args()
    resolved = resolve_config(args.config_file)
    enter_toolbox()
    sys.argv = [str(TOOLBOX / "main.py"), "--config_file", resolved, *rest]
    runpy.run_path(str(TOOLBOX / "main.py"), run_name="__main__")

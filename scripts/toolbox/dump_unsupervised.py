"""Phase 2 - run rPPG-Toolbox's classical methods and save per-video pulse signals (toolbox environment).

    python scripts/toolbox/dump_unsupervised.py --config_file configs/toolbox/UBFC-rPPG_UNSUPERVISED.yaml

The toolbox's own unsupervised mode only prints dataset-level averages. This
script uses exactly the same toolbox preprocessing (Haar-cascade face crop,
72x72 resize) and the same method implementations, but writes one .npz per
video with every method's raw pulse signal and the reference BVP, so that

  * our environment re-scores them with our frozen protocol (score_predictions.py),
  * results can be compared video-by-video with our implementation, and
  * MMPD results can be broken down by condition (Phase 4).

Set DO_PREPROCESS: True in the config the first time (it caches the face crops).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from toolbox_env import REPO, enter_toolbox, resolve_config


def load_methods():
    from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
    from unsupervised_methods.methods.GREEN import GREEN
    from unsupervised_methods.methods.ICA_POH import ICA_POH
    from unsupervised_methods.methods.LGI import LGI
    from unsupervised_methods.methods.OMIT import OMIT
    from unsupervised_methods.methods.PBV import PBV
    from unsupervised_methods.methods.POS_WANG import POS_WANG

    return {
        "POS": lambda x, fs: POS_WANG(x, fs),
        "CHROM": lambda x, fs: CHROME_DEHAAN(x, fs),
        "ICA": lambda x, fs: ICA_POH(x, fs),
        "GREEN": lambda x, fs: GREEN(x),
        "LGI": lambda x, fs: LGI(x),
        "PBV": lambda x, fs: PBV(x),
        "OMIT": lambda x, fs: OMIT(x),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--out", help="output directory (default: cache/toolbox/predictions/<DATASET>)")
    args = ap.parse_args()

    resolved = resolve_config(args.config_file)
    enter_toolbox()
    from config import get_config
    from dataset import data_loader

    config = get_config(argparse.Namespace(config_file=resolved))
    data_cfg = config.UNSUPERVISED.DATA
    loaders = {
        "UBFC-rPPG": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
        "PURE": data_loader.PURELoader.PURELoader,
        "MMPD": data_loader.MMPDLoader.MMPDLoader,
    }
    dataset = loaders[data_cfg.DATASET](name="unsupervised", data_path=data_cfg.DATA_PATH, config_data=data_cfg)
    methods = load_methods()
    chosen = list(config.UNSUPERVISED.METHOD)
    out_dir = Path(args.out) if args.out else REPO / "cache" / "toolbox" / "predictions" / data_cfg.DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = float(data_cfg.FS)

    print(f"{data_cfg.DATASET}: {len(dataset)} videos, methods {chosen} -> {out_dir}")
    for i in range(len(dataset)):
        frames, label, filename, chunk_id = dataset[i]
        frames = frames[..., :3]
        result = {"gt_bvp": np.asarray(label, dtype=np.float64), "fs": fs, "filename": filename, "chunk": chunk_id,
                  "dataset": data_cfg.DATASET}
        for name in chosen:
            t0 = time.perf_counter()
            bvp = np.asarray(methods[name](frames, fs), dtype=np.float64).reshape(-1)
            result[f"pred_{name}"] = bvp
            result[f"ms_per_frame_{name}"] = 1000 * (time.perf_counter() - t0) / len(frames)
        stem = filename if str(chunk_id) == "0" else f"{filename}_{chunk_id}"
        np.savez_compressed(out_dir / f"{stem}.npz", **result)
        print(f"  {stem}: {len(frames)} frames")


if __name__ == "__main__":
    main()

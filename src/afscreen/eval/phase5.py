"""Phase 5 - comparison with baselines, ablation, patch count vs face size, and efficiency.

    ablation      every variant (GREEN / CHROM / POS on one region, then the ingredients added one
                  at a time) on the UBFC videos: heart rate, beats, RMSSD (videos with a
                  reference) and AF false alarms (all 12 normal-rhythm videos), each from the same
                  downstream steps and the same AF classifier, so differences come from pulse
                  extraction and fusion alone
    patch count   6 / 12 / 25 patches and the adaptive choice, on the native videos and on the
                  same videos re-extracted at smaller frame sizes (smaller faces)
    efficiency    ms per frame on this CPU: face landmarks, patch measurement, and the rest
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from afscreen.eval.metrics import beat_f1, ibi_error_ms
from afscreen.eval.reference import align_reference, reference_pulse
from afscreen.pipeline import ALL_VARIANTS, analyse, window_hr
from afscreen.signal.beats import detect_beats
from afscreen.signal.hrv import rmssd_ms
from afscreen.traces import PatchTraces


def _files(dirs) -> list[Path]:
    return sorted({f for d in dirs for f in Path(d).glob("*.npz")})


def video_scores(tr: PatchTraces, variant: str, level: int | None = None) -> dict:
    """Heart-rate, beat and RMSSD errors of one video against its (aligned) reference."""
    from afscreen.signal.spectrum import hr_fft

    r = analyse(tr, variant, level)
    ref = reference_pulse(tr.ref_pulse, tr.fps)
    aligned, _, _ = align_reference(ref, r.pulse, tr.fps)
    rb = detect_beats(aligned, tr.fps)
    ref_hr = hr_fft(ref, tr.fps)
    return {"level": r.level, "abs_err": abs(r.hr_bpm - ref_hr),
            "win_mae": float(np.mean(np.abs(r.window_hr - window_hr(ref, tr.fps, r.window_t)))),
            "ibi_err_ms": ibi_error_ms(r.beats, rb), "beat_f1": beat_f1(r.beats, rb),
            "rmssd_err_ms": abs(rmssd_ms(np.diff(r.beats)) - rmssd_ms(np.diff(rb))),
            "confidence": r.mean_confidence}


def ablation(ref_dirs, all_dirs, model_path="results/phase3/mimic/af_classifier.joblib",
             extra_beats_dir="results/phase4/mitbih", variants=tuple(ALL_VARIANTS)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-video rows, summary per variant)."""
    from afscreen.af.screen import load_model, screen_pulse
    from afscreen.flags.experiment import chosen_rule

    bundle = load_model(model_path)
    rule = chosen_rule(extra_beats_dir) if Path(extra_beats_dir, "chosen_removal.txt").exists() else None
    rows = []
    for f in _files(all_dirs):
        tr = PatchTraces.load(f)
        for v in variants:
            row = {"video": tr.meta.get("id", f.stem), "variant": v, "has_ref": tr.ref_pulse is not None}
            if tr.ref_pulse is not None and Path(f).parent in {Path(d) for d in ref_dirs}:
                row |= video_scores(tr, v)
            w = screen_pulse(analyse(tr, v).pulse, tr.fps, bundle, ectopy=rule)
            rel = w[w.reliable]
            row |= {"af_windows": len(w), "af_reliable": len(rel), "af_false_windows": int((rel.p_af >= 0.5).sum()),
                    "af_false_video": bool(len(rel) and rel.p_af.mean() >= 0.5)}
            rows.append(row)
    per = pd.DataFrame(rows)
    summ = per.groupby("variant", sort=False).agg(
        HR_MAE=("abs_err", "mean"), win_MAE=("win_mae", "mean"), IBI_err_ms=("ibi_err_ms", "mean"),
        beat_F1=("beat_f1", "mean"), RMSSD_err_ms=("rmssd_err_ms", "mean"),
        AF_false_windows=("af_false_windows", "sum"), AF_windows=("af_reliable", "sum"),
        AF_false_videos=("af_false_video", "sum"), videos=("video", "size")).reset_index()
    return per, summ


def patch_count(size_dirs: dict[str, list[str]], levels=(6, 12, 25, None)) -> pd.DataFrame:
    """Rows per (frame size, level, video); level None = adaptive (by face size)."""
    from afscreen.face.patches import choose_level

    rows = []
    for size, dirs in size_dirs.items():
        for f in _files(dirs):
            tr = PatchTraces.load(f)
            if tr.ref_pulse is None:
                continue
            px = float(np.median(tr.counts[(tr.counts > 0).any(axis=1)].sum(axis=1)))
            for lvl in levels:
                s = video_scores(tr, "full", lvl)
                used = s.pop("level")
                rows.append({**s, "frame_size": size, "video": tr.meta.get("id", f.stem),
                             "level": "adaptive" if lvl is None else str(lvl), "used_level": used,
                             "skin_px": px, "auto_level": choose_level(tr.counts)})
    return pd.DataFrame(rows)


def efficiency(video: str, n_frames: int = 300, max_side: int | None = None) -> dict:
    """ms per frame for each stage, on the first ``n_frames`` frames of ``video``."""
    from afscreen.face.patches import measure
    from afscreen.face.tracker import FaceTracker
    from afscreen.io.datasets import downscale, iter_video, video_fps

    fps = video_fps(video)
    frames = []
    for i, fr in enumerate(downscale(iter_video(video), max_side)):
        if i >= n_frames:
            break
        frames.append(fr)
    t_track = t_measure = 0.0
    rgb, counts, facing, pose = [], [], [], []
    with FaceTracker() as tracker:
        for i, fr in enumerate(frames):
            t0 = time.perf_counter()
            face = tracker.detect(fr, i / fps)
            t1 = time.perf_counter()
            c, n, fa = measure(fr, face.points)
            t2 = time.perf_counter()
            t_track += t1 - t0
            t_measure += t2 - t1
            rgb.append(c)
            counts.append(n)
            facing.append(fa)
            pose.append(face.pose)
    tr = PatchTraces(fps=fps, rgb=np.array(rgb), counts=np.array(counts), facing=np.array(facing), pose=np.array(pose))
    t0 = time.perf_counter()
    analyse(tr, "full")
    t_rest = time.perf_counter() - t0
    k = len(frames)
    h, w = frames[0].shape[:2]
    return {"video": Path(video).name, "frames": k, "resolution": f"{w}x{h}", "landmarks_ms": 1000 * t_track / k,
            "patches_ms": 1000 * t_measure / k, "fusion_and_rest_ms": 1000 * t_rest / k,
            "total_ms": 1000 * (t_track + t_measure + t_rest) / k,
            "max_fps": k / (t_track + t_measure + t_rest)}

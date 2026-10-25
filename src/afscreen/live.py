"""Phase 6 - live demo: webcam (or a video file played as if live) with heart rate, rhythm status,
heart-rate flags, RMSSD and confidence, and the patch trust map drawn on the face.

    afscreen live                          webcam 0
    afscreen live --video clip.avi         a file, played at its own frame rate
    afscreen live --video clip.avi --headless --record out.mp4   no window (prints status lines)

Every frame: face landmarks and patch colours go into a rolling 32 s buffer (the 30 s rhythm window
plus a margin) with their real timestamps. About once a second a background thread resamples the
buffer to a steady frame rate and runs the full method on it (so the video does not stutter):

    heart rate      last 8 s; from beats when the rhythm is irregular (the FFT peak is not valid then)
    confidence      of the latest window (patch consistency, beat shape, motion, enough patches)
    rhythm          AF classifier (trained on MIMIC) on the latest 30 s, extra beats removed first,
                    only when the window passes the beat-shape gate; else "signal unreliable"
    RMSSD           latest 30 s (the camera overestimates it about 2x: see Phase 2)
    slow / fast     only after 2 minutes of confident windows, as the plan requires

A screening aid, not a diagnosis: the screen says so.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from afscreen.face.patches import N_FINE

ANALYSIS_FS = 30.0
MIN_SECONDS_HR = 10.0  # heart rate is shown once this much signal is buffered
AF_WINDOW_S = 30.0
FLAG_MIN_CONFIDENT_S = 120.0
MIN_CONFIDENCE = 0.5  # windows below this do not count towards the slow / fast flags


@dataclass
class Status:
    seconds: float = 0.0  # signal buffered
    face: bool = False
    hr_bpm: float = float("nan")
    hr_method: str = ""
    confidence: float = float("nan")
    rhythm: str = "collecting data"
    p_af: float = float("nan")
    rmssd_ms: float = float("nan")
    extra_beats: int = -1  # in the latest 30 s (isolated ones, as removed before the AF check); -1 = unknown
    hr_flag: str = ""
    confident_s: float = 0.0  # confident time accumulated for the slow / fast flags
    level: int = 0
    weights: np.ndarray | None = None
    analysis_ms: float = 0.0


@dataclass
class _Buffer:
    seconds: float
    t: deque = field(default_factory=deque)
    rgb: deque = field(default_factory=deque)
    counts: deque = field(default_factory=deque)
    facing: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)

    def add(self, t, rgb, counts, facing, pose):
        for q, v in ((self.t, t), (self.rgb, rgb), (self.counts, counts), (self.facing, facing), (self.pose, pose)):
            q.append(v)
        while self.t and self.t[-1] - self.t[0] > self.seconds:
            for q in (self.t, self.rgb, self.counts, self.facing, self.pose):
                q.popleft()


def resample(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample samples ``x`` (N, ...) taken at irregular times ``t`` onto a grid at ``fs``.

    Samples that are NaN (no face) stay NaN: a grid point is NaN when its nearest sample is.
    """
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    flat = x.reshape(len(x), -1)
    out = np.full((len(grid), flat.shape[1]), np.nan)
    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(nearest - 1, 0, len(t) - 1)
    nearest = np.where(np.abs(t[prev] - grid) < np.abs(t[nearest] - grid), prev, nearest)
    for c in range(flat.shape[1]):
        ok = np.isfinite(flat[:, c])
        if ok.sum() >= 2:
            out[:, c] = np.interp(grid, t[ok], flat[ok, c])
    out[~np.isfinite(flat[nearest]).all(axis=1)] = np.nan
    return grid, out.reshape((len(grid),) + x.shape[1:])


class LiveMonitor:
    """The demo's logic without the camera or the window, so it can be tested and reused.

    Feed it per-frame patch measurements (``add``) or frames (``add_frame``); call ``update`` to
    analyse the buffer and get a :class:`Status`.
    """

    def __init__(self, buffer_s: float = AF_WINDOW_S + 2.0, model_path="results/phase3/mimic/af_classifier.joblib",
                 extra_beats_dir="results/phase4/mitbih"):
        self.buf = _Buffer(buffer_s)
        self.bundle = None
        if Path(model_path).exists():
            from afscreen.af.screen import load_model

            self.bundle = load_model(model_path)
        self.rule = None
        if Path(extra_beats_dir, "chosen_removal.txt").exists():
            from afscreen.flags.experiment import chosen_rule

            self.rule = chosen_rule(extra_beats_dir)
        self.confident_hr: list[float] = []  # (one per analysed second) heart rates of confident windows
        self.last_face_t = -np.inf
        self._tracker = None

    # ------------------------------------------------------------------ input
    def add(self, t: float, rgb, counts, facing, pose) -> None:
        self.buf.add(t, np.asarray(rgb, float), np.asarray(counts, float), np.asarray(facing, float),
                     np.asarray(pose, float))
        if np.isfinite(np.asarray(pose, float)).all():
            self.last_face_t = t

    def add_frame(self, frame_rgb: np.ndarray, t: float):
        """Track the face in one RGB frame and buffer its patch measurements; returns the face or None."""
        from afscreen.face.patches import measure
        from afscreen.face.tracker import FaceTracker

        if self._tracker is None:
            self._tracker = FaceTracker()
        face = self._tracker.detect(frame_rgb, t)
        if face is None:
            self.add(t, np.full((N_FINE, 3), np.nan), np.zeros(N_FINE), np.full(N_FINE, np.nan), np.full(6, np.nan))
        else:
            self.add(t, *measure(frame_rgb, face.points), face.pose)
        return face

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def reset(self) -> None:
        self.buf = _Buffer(self.buf.seconds)
        self.confident_hr = []

    # ------------------------------------------------------------------ analysis
    def snapshot(self):
        """The buffer as a :class:`PatchTraces` at ANALYSIS_FS, or None if too short."""
        from afscreen.traces import PatchTraces

        if len(self.buf.t) < 2:
            return None
        t = np.array(self.buf.t)
        if t[-1] - t[0] < MIN_SECONDS_HR:
            return None
        _, rgb = resample(t, np.array(self.buf.rgb), ANALYSIS_FS)
        _, counts = resample(t, np.array(self.buf.counts), ANALYSIS_FS)
        _, facing = resample(t, np.array(self.buf.facing), ANALYSIS_FS)
        _, pose = resample(t, np.array(self.buf.pose), ANALYSIS_FS)
        return PatchTraces(fps=ANALYSIS_FS, rgb=rgb, counts=np.nan_to_num(counts), facing=facing, pose=pose)

    def update(self) -> Status:
        """Snapshot the buffer and analyse it (both in the calling thread)."""
        return self.analyse_snapshot(*self.take_snapshot())

    def take_snapshot(self):
        """(traces or None, seconds buffered, face seen in the last second) - call from the thread
        that adds frames; the result can then be analysed in another thread."""
        seconds = (self.buf.t[-1] - self.buf.t[0]) if len(self.buf.t) > 1 else 0.0
        face = bool(self.buf.t) and self.buf.t[-1] - self.last_face_t < 1.0
        return self.snapshot(), seconds, face

    def analyse_snapshot(self, tr, seconds: float, face: bool) -> Status:
        from afscreen.af.features import nrmssd
        from afscreen.pipeline import analyse
        from afscreen.signal.hrv import rmssd_ms

        t0 = time.perf_counter()
        st = Status(seconds=seconds, face=face)
        if tr is None or tr.detection_rate < 0.5:
            st.rhythm = "collecting data" if st.face else "no face"
            return st
        try:
            r = analyse(tr, "full")
        except ValueError:  # e.g. no patch visible long enough
            st.rhythm = "signal unreliable"
            return st
        fs = tr.fps
        st.level, st.weights = r.level, r.fusion.weights[-1]
        st.confidence = float(r.confidence[-1])
        recent = r.beats[r.beats >= r.beats[-1] - AF_WINDOW_S] if len(r.beats) else r.beats
        irregular = len(recent) >= 6 and nrmssd(np.diff(recent)) > 0.1
        last8 = r.beats[r.beats >= tr.n_frames / fs - 8.0]
        if irregular and len(last8) >= 3:
            st.hr_bpm, st.hr_method = 60.0 / float(np.mean(np.diff(last8))), "beats (irregular rhythm)"
        else:
            st.hr_bpm, st.hr_method = float(r.window_hr[-1]), "spectrum"
        st.rmssd_ms = rmssd_ms(np.diff(recent))

        # Rhythm: the latest 30 s window, gated.
        n = int(AF_WINDOW_S * fs)
        if self.bundle is None:
            st.rhythm = "no AF model (run afscreen af-train)"
        elif len(r.pulse) < n:
            st.rhythm = f"collecting data ({AF_WINDOW_S - len(r.pulse) / fs:.0f} s to first rhythm check)"
        else:
            from afscreen.af.screen import screen_pulse

            w = screen_pulse(r.pulse[-n:], fs, self.bundle, ectopy=self.rule).iloc[-1]
            st.p_af = float(w.p_af)
            if self.rule is not None:
                from afscreen.flags.ectopy import ibis_without_extra_beats
                from afscreen.signal.beats import detect_beats

                _, n_extra, removed = ibis_without_extra_beats(detect_beats(r.pulse[-n:], fs), self.rule)
                st.extra_beats = n_extra if removed else 0
            if not w.reliable:
                st.rhythm = "signal unreliable"
            else:
                st.rhythm = "possible AF - see a doctor for an ECG" if w.p_af >= 0.5 else "regular rhythm"

        # Slow / fast: median of confident heart rates over at least 2 minutes.
        if st.confidence >= MIN_CONFIDENCE and np.isfinite(st.hr_bpm):
            self.confident_hr.append(st.hr_bpm)
        st.confident_s = float(len(self.confident_hr))  # one entry per analysed second
        if st.confident_s >= FLAG_MIN_CONFIDENT_S:
            from afscreen.flags.experiment import hr_flag

            med = float(np.median(self.confident_hr[-int(FLAG_MIN_CONFIDENT_S):]))
            flag = hr_flag(med)
            st.hr_flag = {"slow": f"slow heart rate ({med:.0f} BPM): can be normal for fit people; "
                                  "see a doctor if dizzy or faint",
                          "fast": f"fast heart rate at rest ({med:.0f} BPM)",
                          "normal": f"resting rate normal ({med:.0f} BPM)"}.get(flag, "")
        st.analysis_ms = 1000 * (time.perf_counter() - t0)
        return st


# ---------------------------------------------------------------------- drawing


def draw_status(bgr: np.ndarray, st: Status) -> np.ndarray:
    """Text panel along the bottom of the frame."""
    import cv2

    h, w = bgr.shape[:2]
    panel_h = 118
    out = cv2.copyMakeBorder(bgr, 0, panel_h, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    y = h + 22
    hr = "--" if not np.isfinite(st.hr_bpm) else f"{st.hr_bpm:.0f}"
    conf = "--" if not np.isfinite(st.confidence) else f"{st.confidence:.2f}"
    rm = "--" if not np.isfinite(st.rmssd_ms) else f"{st.rmssd_ms:.0f} ms"
    colour = {"regular rhythm": (90, 200, 90), "signal unreliable": (0, 165, 255)}.get(st.rhythm, (230, 230, 230))
    if st.rhythm.startswith("possible AF"):
        colour = (60, 60, 255)
    extra = f"   extra beats (30 s): {st.extra_beats}" if st.extra_beats >= 0 else ""
    lines = [(f"Heart rate {hr} BPM   confidence {conf}   RMSSD {rm} (camera overestimates)", (240, 240, 240)),
             (f"Rhythm: {st.rhythm}{extra}", colour),
             (st.hr_flag or f"slow / fast check after 2 min of confident signal ({st.confident_s:.0f} s so far)",
              (200, 200, 200)),
             ("Screening aid, not a diagnosis.  q: quit   r: reset", (150, 150, 150))]
    for text, c in lines:
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        y += 25
    if np.isfinite(st.confidence):  # confidence bar at the right
        x0, x1 = w - 110, w - 10
        cv2.rectangle(out, (x0, h + 10), (x1, h + 22), (80, 80, 80), -1)
        cv2.rectangle(out, (x0, h + 10), (x0 + int(100 * np.clip(st.confidence, 0, 1)), h + 22), (90, 200, 90), -1)
    if not st.face:
        cv2.putText(out, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------- the loop


def run(camera: int = 0, video: str | None = None, headless: bool = False, record: str | None = None,
        max_seconds: float | None = None, max_side: int = 640, print_every: float = 5.0) -> list[Status]:
    """Run the demo; returns the status after each analysis (useful when headless)."""
    import cv2

    from afscreen.face import patches

    cap = cv2.VideoCapture(video if video else camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {'video ' + video if video else f'camera {camera}'}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) if video else 0.0
    file_fps = file_fps if file_fps and file_fps > 1 else 30.0
    mon = LiveMonitor()
    status, history = Status(), []
    lock, worker = threading.Lock(), None
    writer = None
    start, i, last_print, next_analysis = time.perf_counter(), 0, -np.inf, 0.0

    def analyse(snap):
        nonlocal status
        st = mon.analyse_snapshot(*snap)
        with lock:
            status = st
            history.append(st)

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / file_fps if video else time.perf_counter() - start
            if max_seconds is not None and t > max_seconds:
                break
            hgt, wid = bgr.shape[:2]
            if max(hgt, wid) > max_side:
                s = max_side / max(hgt, wid)
                bgr = cv2.resize(bgr, (round(wid * s), round(hgt * s)), interpolation=cv2.INTER_AREA)
            face = mon.add_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t)
            # Analyse about once a second. Live: in a background thread on a snapshot, skipping a
            # turn while the previous analysis still runs. A file played headless: in this thread,
            # so every second of the file is analysed (repeatable results).
            if t >= next_analysis:
                if video and headless:
                    analyse(mon.take_snapshot())
                    next_analysis = t + 1.0
                elif worker is None or not worker.is_alive():
                    worker = threading.Thread(target=analyse, args=(mon.take_snapshot(),), daemon=True)
                    worker.start()
                    next_analysis = t + 1.0
            with lock:
                st = status
            if face is not None and st.weights is not None and len(st.weights) == st.level:
                bgr = patches.draw(bgr, face.points, st.level, st.weights)
            st.face = face is not None
            frame = draw_status(bgr, st)
            if record:
                if writer is None:
                    writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), file_fps if video else 30.0,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if headless:
                if t - last_print >= print_every and np.isfinite(st.hr_bpm):
                    print(f"t={t:5.1f}s  HR {st.hr_bpm:5.1f} ({st.hr_method})  conf {st.confidence:.2f}  "
                          f"rhythm: {st.rhythm}  extra beats {st.extra_beats if st.extra_beats >= 0 else '-'}  "
                          f"RMSSD {st.rmssd_ms:.0f} ms  analysis {st.analysis_ms:.0f} ms",
                          flush=True)
                    last_print = t
            else:
                cv2.imshow("afscreen live", frame)
                key = cv2.waitKey(1 if not video else max(1, int(1000 / file_fps))) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    mon.reset()
            i += 1
    finally:
        if worker is not None:
            worker.join(timeout=5)
        cap.release()
        mon.close()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    return history

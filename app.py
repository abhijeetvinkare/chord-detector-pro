import re
import sys
import time
import uuid
import hashlib
import shutil
import tempfile
import webbrowser
import threading
import subprocess
from pathlib import Path

import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response

try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

app = FastAPI(title="Chord Detector Offline")

BASE_DIR = Path(__file__).resolve().parent
CACHE_AUDIO_DIR = BASE_DIR / "cached_audio"
CACHE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "chord_detector_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="16" fill="#f97316"/>
  <path d="M26 43V21l20-4v22" fill="none" stroke="#111827" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="21" cy="43" r="6" fill="#111827"/>
  <circle cx="41" cy="39" r="6" fill="#111827"/>
</svg>"""

JOBS_PROGRESS = {}


def update_job(job_id: str, **kwargs):
    if not job_id:
        return
    if job_id not in JOBS_PROGRESS:
        JOBS_PROGRESS[job_id] = {
            "percent": 0,
            "stage": "Starting...",
            "detail": "",
            "eta": "--",
            "speed": "--",
            "step": 1,
            "done": False,
        }
    JOBS_PROGRESS[job_id].update(kwargs)


def compute_short_id(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()[:24]


# ==============================================================================
# 1. YOUTUBE / URL AUDIO DOWNLOADER (WITH LIVE PROGRESS)
# ==============================================================================
def download_audio_from_url(url: str, dest_dir: Path, job_id: str = "") -> tuple[Path, str]:
    if not YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp is not installed. Run 'pip install yt-dlp' in your venv first.")

    update_job(
        job_id,
        percent=4,
        step=1,
        stage="Fetching Link Metadata...",
        detail=f"Connecting to {url[:60]}...",
        eta="0:12",
        speed="Connecting...",
    )

    def progress_hook(d):
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            pct = int((downloaded / total) * 12) + 4 if total > 0 else 8
            speed_bps = d.get("speed") or 0
            speed_str = f"{(speed_bps / (1024 * 1024)):.1f} MB/s" if speed_bps > 0 else "Downloading..."
            eta_sec = d.get("eta")
            eta_str = f"0:{int(eta_sec):02d}" if isinstance(eta_sec, (int, float)) and eta_sec < 60 else "--"
            update_job(
                job_id,
                percent=min(16, max(4, pct)),
                step=1,
                stage="Downloading Audio Stream...",
                detail=f"Downloaded {(downloaded / (1024 * 1024)):.2f} MB",
                eta=eta_str,
                speed=speed_str,
            )
        elif d.get("status") == "finished":
            update_job(
                job_id,
                percent=17,
                step=1,
                stage="Audio Downloaded!",
                detail="Preparing audio for chord analysis...",
                eta="0:05",
                speed="Done",
            )

    out_tmpl = str(dest_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or "YouTube Audio"
        downloaded_path = Path(ydl.prepare_filename(info))
        if not downloaded_path.exists():
            matches = list(dest_dir.glob(f"{info.get('id', '*')}*"))
            if not matches:
                raise RuntimeError("Could not locate downloaded audio file.")
            downloaded_path = matches[0]
        return downloaded_path, title


# ==============================================================================
# 2. MUSIC THEORY, KEY PROFILES, BPM & METER-AWARE SUBDIVISION
# ==============================================================================
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def build_chord_templates():
    labels = []
    templates = []
    chord_types = {
        "Major": [(0, 1.0), (4, 0.85), (7, 0.90)],
        "Minor": [(0, 1.0), (3, 0.85), (7, 0.90)],
    }
    for root_idx, root_name in enumerate(PITCH_CLASSES):
        for quality, intervals in chord_types.items():
            vec = np.zeros(12, dtype=np.float32)
            for interval, weight in intervals:
                vec[(root_idx + interval) % 12] = weight
            vec /= np.linalg.norm(vec)
            labels.append(f"{root_name} {quality}")
            templates.append(vec)
    return labels, np.array(templates)


CHORD_LABELS, CHORD_TEMPLATES = build_chord_templates()


def estimate_musical_key(chroma_mean: np.ndarray) -> str:
    best_score = -np.inf
    best_key = "C Major"
    for idx, root in enumerate(PITCH_CLASSES):
        maj_corr = np.corrcoef(chroma_mean, np.roll(MAJOR_KEY_PROFILE, idx))[0, 1]
        min_corr = np.corrcoef(chroma_mean, np.roll(MINOR_KEY_PROFILE, idx))[0, 1]
        if maj_corr > best_score:
            best_score = maj_corr
            best_key = f"{root} Major"
        if min_corr > best_score:
            best_score = min_corr
            best_key = f"{root} Minor"
    return best_key


def get_diatonic_chord_set(key_label: str) -> set[str]:
    parts = key_label.split(" ")
    root = parts[0]
    qual = parts[1] if len(parts) > 1 else "Major"
    r_idx = PITCH_CLASSES.index(root) if root in PITCH_CLASSES else 0
    if qual == "Minor":
        maj_idx = (r_idx + 3) % 12
    else:
        maj_idx = r_idx

    return {
        f"{PITCH_CLASSES[maj_idx]} Major",
        f"{PITCH_CLASSES[(maj_idx + 5) % 12]} Major",
        f"{PITCH_CLASSES[(maj_idx + 7) % 12]} Major",
        f"{PITCH_CLASSES[(maj_idx + 9) % 12]} Minor",
        f"{PITCH_CLASSES[(maj_idx + 2) % 12]} Minor",
        f"{PITCH_CLASSES[(maj_idx + 4) % 12]} Minor",
    }


def estimate_time_signature(onset_env: np.ndarray, beat_frames: np.ndarray) -> str:
    if len(beat_frames) < 12:
        return "4/4"

    valid_beats = beat_frames[beat_frames < len(onset_env)]
    if len(valid_beats) < 12:
        return "4/4"

    beat_env = onset_env[valid_beats].astype(np.float32)
    beat_env -= np.mean(beat_env)
    norm = np.linalg.norm(beat_env)
    if norm == 0:
        return "4/4"
    beat_env /= norm

    ac = np.correlate(beat_env, beat_env, mode="full")
    mid = len(ac) // 2
    ac_pos = ac[mid:]

    score_3 = float(ac_pos[3]) if len(ac_pos) > 3 else 0.0
    score_4 = float(ac_pos[4]) if len(ac_pos) > 4 else 0.0
    score_6 = float(ac_pos[6]) if len(ac_pos) > 6 else 0.0

    sub_triplet_energy = 0.0
    sub_duple_energy = 0.0
    for i in range(min(32, len(valid_beats) - 1)):
        b0, b1 = int(valid_beats[i]), int(valid_beats[i + 1])
        span = b1 - b0
        if span >= 6:
            t1 = b0 + span // 3
            t2 = b0 + (2 * span) // 3
            d1 = b0 + span // 2
            sub_triplet_energy += 0.5 * (onset_env[t1] + onset_env[t2])
            sub_duple_energy += onset_env[d1]

    if sub_triplet_energy > sub_duple_energy * 1.18 and score_6 > score_4 * 1.05:
        return "6/8"
    if score_3 > score_4 * 1.15 and score_3 > 0.12:
        return "3/4"
    return "4/4"


def estimate_accurate_bpm_and_beats(y_full: np.ndarray, sr: int, hop_length: int, forced_time_sig: str = "auto"):
    y_perc = librosa.effects.percussive(y_full)
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)

    tempo_raw, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=90.0,
        tightness=100,
    )
    bpm = float(np.atleast_1d(tempo_raw)[0])

    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tg_mean = np.mean(tempogram, axis=1)
    tempo_freqs = librosa.tempo_frequencies(tempogram.shape[0], sr=sr, hop_length=hop_length)

    def strength_at(target_bpm: float) -> float:
        if target_bpm < 40 or target_bpm > 240:
            return 0.0
        idx = int(np.argmin(np.abs(tempo_freqs - target_bpm)))
        lo = max(1, idx - 1)
        hi = min(len(tg_mean), idx + 2)
        return float(np.max(tg_mean[lo:hi]))

    s_1x = strength_at(bpm)
    s_half = strength_at(bpm * 0.5)
    s_double = strength_at(bpm * 2.0)

    if bpm > 138.0 and s_half >= 0.72 * s_1x:
        bpm *= 0.5
    elif bpm < 66.0 and s_double >= 0.72 * s_1x:
        bpm *= 2.0

    if forced_time_sig in ("4/4", "3/4", "6/8"):
        time_sig = forced_time_sig
    else:
        time_sig = estimate_time_signature(onset_env, beat_frames)

    return round(bpm, 1), beat_frames, onset_env, time_sig


def build_meter_aware_grid(beat_frames: np.ndarray, onset_env: np.ndarray, total_frames: int, time_sig: str):
    if len(beat_frames) <= 4:
        grid = np.arange(0, total_frames, 8)
        if grid[-1] != total_frames:
            grid = np.append(grid, total_frames)
        downbeat_mask = np.zeros(len(grid) - 1, dtype=bool)
        downbeat_mask[::4] = True
        return grid, downbeat_mask

    beats_per_bar = 3 if time_sig == "3/4" else (2 if time_sig == "6/8" else 4)
    valid_b = beat_frames[beat_frames < len(onset_env)]
    best_offset = 0
    if len(valid_b) >= beats_per_bar * 2:
        phase_scores = []
        for off in range(beats_per_bar):
            phase_scores.append(float(np.mean(onset_env[valid_b[off::beats_per_bar]])))
        best_offset = int(np.argmax(phase_scores))

    pts = []
    is_downbeat_step = []

    if int(beat_frames[0]) > 0:
        pts.append(0)
        is_downbeat_step.append(False)

    subdiv = 3 if time_sig == "6/8" else 2
    for i in range(len(beat_frames) - 1):
        b0 = int(beat_frames[i])
        b1 = int(beat_frames[i + 1])
        is_bar_start = ((i - best_offset) % beats_per_bar == 0)

        for s in range(subdiv):
            pos = b0 + ((b1 - b0) * s) // subdiv
            if not pts or pos > pts[-1]:
                pts.append(pos)
                is_downbeat_step.append(is_bar_start and s == 0)

    last_b = int(beat_frames[-1])
    if not pts or last_b > pts[-1]:
        pts.append(last_b)
        is_downbeat_step.append(((len(beat_frames) - 1 - best_offset) % beats_per_bar == 0))

    if pts[-1] < total_frames:
        pts.append(total_frames)

    grid = np.array(pts, dtype=np.int32)
    mask = np.array(is_downbeat_step[: max(1, len(grid) - 1)], dtype=bool)
    return grid, mask


def decode_meter_aware_viterbi(probs: np.ndarray, downbeat_mask: np.ndarray) -> np.ndarray:
    n_states, n_steps = probs.shape
    if n_steps == 0:
        return np.array([], dtype=np.int32)

    eps = 1e-12
    log_probs = np.log(np.clip(probs, eps, 1.0))

    dp = np.zeros((n_states, n_steps), dtype=np.float32)
    backptr = np.zeros((n_states, n_steps), dtype=np.int32)
    dp[:, 0] = log_probs[:, 0]

    for t in range(1, n_steps):
        is_downbeat = bool(downbeat_mask[t]) if t < len(downbeat_mask) else False
        stay_prob = 0.45 if is_downbeat else 0.76
        switch_prob = (1.0 - stay_prob) / max(1, n_states - 1)

        log_stay = np.log(stay_prob)
        log_switch = np.log(switch_prob)

        prev = dp[:, t - 1]
        best_prev_idx = int(np.argmax(prev))
        best_prev_val = float(prev[best_prev_idx])

        stay_scores = prev + log_stay
        switch_scores = np.full(n_states, best_prev_val + log_switch, dtype=np.float32)

        use_stay = stay_scores >= switch_scores
        dp[:, t] = np.where(use_stay, stay_scores, switch_scores) + log_probs[:, t]
        backptr[:, t] = np.where(use_stay, np.arange(n_states, dtype=np.int32), best_prev_idx)

    states = np.zeros(n_steps, dtype=np.int32)
    states[-1] = int(np.argmax(dp[:, -1]))
    for t in range(n_steps - 1, 0, -1):
        states[t - 1] = backptr[states[t], t]
    return states


def isolate_harmonic_stems(audio_path: str, work_dir: str, job_id: str = "") -> str:
    cmd = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "-o", work_dir, audio_path]
    update_job(
        job_id,
        percent=18,
        step=2,
        stage="Demucs AI: Separating Stems...",
        detail="Loading htdemucs neural model (isolating Bass & Instruments from Vocals/Drums)...",
        eta="Calculating...",
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    last_lines = []
    pct_pattern = re.compile(r"(\d{1,3})%\|")
    eta_pattern = re.compile(r"<([0-9:]+),\s*([0-9.]+[a-zA-Z/]+)")

    if proc.stdout:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            last_lines.append(line)
            if len(last_lines) > 15:
                last_lines.pop(0)

            m_pct = pct_pattern.search(line)
            if m_pct:
                raw_pct = int(m_pct.group(1))
                overall_pct = 18 + int(raw_pct * 0.58)
                m_eta = eta_pattern.search(line)
                eta_str = m_eta.group(1) if m_eta else "Working..."
                speed_str = m_eta.group(2) if m_eta else ""
                update_job(
                    job_id,
                    percent=overall_pct,
                    step=2,
                    stage=f"Demucs AI Separation ({raw_pct}%)",
                    detail=f"Neural stem split: {raw_pct}% complete" + (f" • Speed: {speed_str}" if speed_str else ""),
                    eta=eta_str,
                )
            elif "Downloading" in line:
                update_job(
                    job_id,
                    percent=18,
                    step=2,
                    stage="Downloading Demucs AI Model (One-Time)...",
                    detail=line[:90],
                    eta="~0:30",
                )

    ret = proc.wait()
    if ret != 0:
        err_tail = "\n".join(last_lines[-5:])
        raise RuntimeError(f"Demucs failed (code {ret}): {err_tail}")

    update_job(
        job_id,
        percent=78,
        step=2,
        stage="Blending Harmonic Stems...",
        detail="Combining clean Bass + Guitar/Keys tracks...",
        eta="0:04",
    )

    stem_folder = Path(work_dir) / "htdemucs" / Path(audio_path).stem
    y_bass, sr = librosa.load(str(stem_folder / "bass.wav"), sr=22050, mono=True)
    y_other, _ = librosa.load(str(stem_folder / "other.wav"), sr=22050, mono=True)
    min_len = min(len(y_bass), len(y_other))
    y_harmonic = (0.75 * y_bass[:min_len]) + (1.0 * y_other[:min_len])
    out_wav = str(Path(work_dir) / "harmonic_mix.wav")
    import soundfile as sf
    sf.write(out_wav, y_harmonic, sr)
    return out_wav


# ==============================================================================
# 3. HIGH-RESPONSIVENESS KEY & METER-AWARE CHORD ENGINE
# ==============================================================================
def analyze_audio_file(
    audio_path: str,
    use_demucs: bool = False,
    job_id: str = "",
    forced_time_sig: str = "auto",
):
    print(f"[1/4] Fresh analysis: {audio_path} (Meter={forced_time_sig})")
    sr = 22050
    hop_length = 512

    update_job(
        job_id,
        percent=18,
        step=1,
        stage="Reading Audio Track...",
        detail="Decoding audio waveform...",
        eta="0:25" if use_demucs else "0:03",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        y_orig, sr = librosa.load(audio_path, sr=sr, mono=True)
        duration = float(librosa.get_duration(y=y_orig, sr=sr))

        target_audio = audio_path
        if use_demucs:
            if not DEMUCS_AVAILABLE:
                raise RuntimeError("Demucs is not installed in the active environment.")
            print("[1.5/4] Separating stems with Demucs...")
            target_audio = isolate_harmonic_stems(audio_path, tmp_dir, job_id=job_id)
            y_harm, _ = librosa.load(target_audio, sr=sr, mono=True)
        else:
            y_harm = y_orig

        update_job(
            job_id,
            percent=82 if use_demucs else 42,
            step=3,
            stage="Extracting High-Res Chroma...",
            detail="Analyzing 24-bin harmonic & bass frequency spectrum...",
            eta="0:02",
        )

        print("[2/4] Extracting high-res chroma & detecting meter/beats...")
        y_harmonic_only = librosa.effects.harmonic(y_harm)
        chroma_full = librosa.feature.chroma_cqt(
            y=y_harmonic_only, sr=sr, hop_length=hop_length, bins_per_octave=24
        )
        chroma_bass = librosa.feature.chroma_cqt(
            y=y_harmonic_only, sr=sr, hop_length=hop_length,
            fmin=librosa.note_to_hz("C1"), n_octaves=3, bins_per_octave=24
        )
        chroma = (0.62 * chroma_full) + (0.38 * chroma_bass)
        global_key = estimate_musical_key(np.mean(chroma, axis=1))
        diatonic_set = get_diatonic_chord_set(global_key)

        bpm, beat_frames, onset_env, time_sig = estimate_accurate_bpm_and_beats(
            y_orig, sr=sr, hop_length=hop_length, forced_time_sig=forced_time_sig
        )

        update_job(
            job_id,
            percent=90 if use_demucs else 72,
            step=3,
            stage=f"Aligning Chords to {time_sig} Bar Grid ({global_key})...",
            detail=f"Locking downbeats & {time_sig} subdivisions at {bpm} BPM...",
            eta="0:01",
        )

        print(f"[3/4] Aligning chords to {time_sig} meter-aware grid...")
        sync_frames, downbeat_mask = build_meter_aware_grid(
            beat_frames, onset_env, chroma.shape[1], time_sig
        )

        chroma_agg = librosa.util.sync(chroma, sync_frames, aggregate=np.mean)
        step_times = librosa.frames_to_time(sync_frames, sr=sr, hop_length=hop_length)

        norms = np.linalg.norm(chroma_agg, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        chroma_norm = chroma_agg / norms

        update_job(
            job_id,
            percent=96 if use_demucs else 90,
            step=4,
            stage=f"Finalizing {time_sig} Chord Timeline...",
            detail="Running downbeat-weighted Viterbi decoding...",
            eta="0:01",
        )

        similarity = np.dot(CHORD_TEMPLATES, chroma_norm)

        diatonic_bonus = np.array(
            [0.065 if lbl in diatonic_set else 0.0 for lbl in CHORD_LABELS],
            dtype=np.float32,
        )[:, None]
        similarity = similarity + diatonic_bonus

        exp_sim = np.exp(similarity * 14.0)
        probs = exp_sim / np.sum(exp_sim, axis=0, keepdims=True)

        state_seq = decode_meter_aware_viterbi(probs, downbeat_mask)

        raw_segments = []
        curr_state = state_seq[0]
        seg_start = float(step_times[0])
        for idx in range(1, len(state_seq)):
            if state_seq[idx] != curr_state:
                seg_end = float(step_times[idx])
                raw_segments.append([seg_start, seg_end, CHORD_LABELS[curr_state]])
                curr_state = state_seq[idx]
                seg_start = seg_end
        raw_segments.append([seg_start, float(step_times[-1]), CHORD_LABELS[curr_state]])

        merged = []
        for seg in raw_segments:
            if merged and ((seg[1] - seg[0]) < 0.32 or merged[-1][2] == seg[2]):
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)

        chords = [
            {"start": round(s, 3), "end": round(e, 3), "chord": c}
            for s, e, c in merged
        ]
        print(f"[4/4] Done! {len(chords)} chords | {global_key} | {bpm} BPM | {time_sig}")
        update_job(
            job_id,
            percent=100,
            step=4,
            stage="Complete!",
            detail=f"Detected {len(chords)} chords • {global_key} • {bpm} BPM • {time_sig}",
            eta="0:00",
            done=True,
        )
        return {
            "chords": chords,
            "key": global_key,
            "bpm": bpm,
            "time_signature": time_sig,
            "duration": duration,
        }


# ==============================================================================
# 4. PRODUCER STUDIO UI (POPPINS LEAD SHEET PRINT + CLEAN DUAL-VIEW)
# ==============================================================================
HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Chord Detector</title>
<link rel="icon" type="image/svg+xml" href="/favicon.ico" />
<style>
  :root {
    --bg: #0f1318;
    --card: #171c23;
    --card-border: #262e38;
    --text: #f1f5f9;
    --muted: #8b96a5;
    --accent: #f97316;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; }

  .container { width: 100%; max-width: 1140px; padding: 20px 24px 104px; }

  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap; gap: 12px; }
  .brand { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .logo-box {
    width: 36px; height: 36px; border-radius: 9px; background: var(--accent);
    display: flex; align-items: center; justify-content: center; font-size: 18px; color: #111; font-weight: bold;
  }
  .brand h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
  .badge {
    font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px;
    background: #1e252e; color: var(--muted); border: 1px solid var(--card-border); letter-spacing: 0.5px;
  }
  .meta-group-top { display: none; align-items: center; gap: 6px; flex-wrap: wrap; }
  .meta-pill {
    font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 6px;
    background: #1b222c; color: #38bdf8; border: 1px solid #283545;
  }
  .sig-pill {
    font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 6px;
    background: #231d36; color: #c084fc; border: 1px solid #4c3575; cursor: pointer; transition: 0.15s;
  }
  .sig-pill:hover { border-color: var(--accent); color: #fff; background: #2d2447; }
  .bpm-mult-btn {
    font-size: 11px; font-weight: 700; padding: 4px 7px; border-radius: 6px;
    background: #1b222c; color: #94a3b8; border: 1px solid #283545; cursor: pointer; transition: 0.15s;
  }
  .bpm-mult-btn:hover { border-color: var(--accent); color: #fff; }

  .toggle-wrap { display: flex; align-items: center; gap: 10px; font-size: 14px; color: var(--muted); cursor: pointer; user-select: none; }
  .switch { position: relative; width: 42px; height: 22px; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: #262e38; border-radius: 999px; transition: 0.2s; }
  .slider:before {
    content: ""; position: absolute; height: 16px; width: 16px; left: 3px; bottom: 3px;
    background: #cbd5e1; border-radius: 50%; transition: 0.2s;
  }
  input:checked + .slider { background: var(--accent); }
  input:checked + .slider:before { transform: translateX(20px); background: #fff; }

  .input-grid {
    display: grid; grid-template-columns: 1.1fr 1fr; gap: 14px; margin-bottom: 18px;
  }
  @media (max-width: 780px) {
    .input-grid { grid-template-columns: 1fr; }
  }
  .dropzone {
    border: 1.5px dashed #323c49; background: #141920; border-radius: 14px;
    padding: 16px 20px; display: flex; align-items: center; justify-content: center; gap: 14px;
    cursor: pointer; transition: 0.2s; user-select: none;
  }
  .dropzone:hover, .dropzone.dragover { border-color: var(--accent); background: #1a2029; }
  .upload-icon {
    width: 38px; height: 38px; border-radius: 10px; background: #1f2630;
    display: flex; align-items: center; justify-content: center; color: var(--accent); font-size: 18px; flex-shrink: 0;
  }
  .drop-title { font-weight: 700; font-size: 14.5px; }
  .drop-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

  .url-box {
    border: 1px solid var(--card-border); background: #141920; border-radius: 14px;
    padding: 14px 16px; display: flex; flex-direction: column; justify-content: center; gap: 8px;
  }
  .url-label { font-size: 12px; font-weight: 700; color: var(--muted); letter-spacing: 0.2px; }
  .url-row { display: flex; gap: 8px; }
  .url-input {
    flex: 1; background: #0f1318; border: 1px solid #2b3441; border-radius: 9px;
    padding: 9px 12px; color: var(--text); font-size: 13.5px; outline: none; transition: 0.15s;
  }
  .url-input:focus { border-color: var(--accent); }
  .url-btn {
    background: var(--accent); color: #111; border: none; border-radius: 9px;
    padding: 0 15px; font-weight: 700; font-size: 13.5px; cursor: pointer; white-space: nowrap; transition: 0.15s;
  }
  .url-btn:hover { filter: brightness(1.08); }

  .stage-card {
    background: var(--card); border: 1px solid var(--card-border); border-radius: 16px;
    padding: 20px 24px 24px; margin-bottom: 18px; position: relative;
  }
  .song-title {
    font-size: 14px; color: var(--muted); margin-bottom: 14px; text-align: center;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .stage-body {
    display: flex; align-items: center; justify-content: space-around; gap: 22px; flex-wrap: wrap;
  }
  .stage-center { text-align: center; min-width: 190px; }
  .chord-short {
    font-size: 104px; font-weight: 800; line-height: 1; letter-spacing: -2px;
    color: #9e9b38; margin-bottom: 8px;
  }
  .chord-full { font-size: 16px; color: #cbd5e1; font-weight: 500; margin-bottom: 12px; }
  .chord-next { font-size: 14px; color: var(--muted); }
  .chord-next strong { color: #f8fafc; font-size: 17px; margin-left: 4px; }

  .diagram-panel {
    background: #12161c; border: 1px solid var(--card-border); border-radius: 12px;
    padding: 12px 16px; display: flex; flex-direction: column; align-items: center; gap: 8px;
    min-width: 168px;
  }
  .inst-tabs {
    display: flex; gap: 4px; background: #191f27; padding: 3px; border-radius: 7px; width: 100%;
  }
  .inst-tab {
    flex: 1; border: none; background: transparent; color: var(--muted); font-size: 11.5px;
    font-weight: 600; padding: 4px 8px; border-radius: 5px; cursor: pointer; transition: 0.15s;
  }
  .inst-tab.active { background: #262f3b; color: #f8fafc; }
  #diagram-container { display: flex; align-items: center; justify-content: center; min-height: 124px; }

  .circle-panel {
    background: #12161c; border: 1px solid var(--card-border); border-radius: 12px;
    padding: 10px 14px 12px; display: flex; flex-direction: column; align-items: center; gap: 6px;
    min-width: 245px;
  }
  .circle-header {
    width: 100%; display: flex; justify-content: space-between; align-items: center;
    font-size: 11.5px; color: var(--muted); font-weight: 700;
  }
  .circle-key-tag {
    color: #38bdf8; background: #19222d; border: 1px solid #263547;
    padding: 2px 7px; border-radius: 5px; font-size: 11px;
  }
  .scale-chips {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 4px; max-width: 235px; margin-top: 2px;
  }
  .scale-chip {
    font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 5px;
    background: #1b222c; color: #cbd5e1; border: 1px solid #2b3544; cursor: pointer; transition: 0.12s;
  }
  .scale-chip:hover { border-color: var(--accent); color: #fff; }
  .scale-chip.tonic {
    background: rgba(249, 115, 22, 0.2); border-color: var(--accent); color: #fb923c;
  }

  .timeline-wrapper {
    background: #13171d; border: 1px solid var(--card-border); border-radius: 12px;
    padding: 10px 0; overflow-x: auto; position: relative; white-space: nowrap;
    scrollbar-width: thin; scrollbar-color: #2d3748 #13171d;
  }
  .timeline-track { display: inline-flex; align-items: center; gap: 4px; padding: 0 45%; min-height: 54px; }
  .chord-block {
    height: 46px; border-radius: 7px; display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13.5px; color: #f8fafc; cursor: pointer;
    padding: 0 8px; user-select: none;
    border: 2px solid transparent; flex-shrink: 0;
  }
  .chord-block:hover { filter: brightness(1.15); }
  .chord-block.active {
    border-color: #ffffff; box-shadow: 0 0 14px rgba(255,255,255,0.28); transform: scale(1.04); z-index: 2;
  }

  .shortcut-hint {
    text-align: center; font-size: 11.5px; color: #64748b; margin-top: 12px;
  }
  .shortcut-hint kbd {
    background: #1b2129; border: 1px solid #2b3440; border-radius: 4px; padding: 1px 6px;
    font-family: inherit; color: #94a3b8;
  }

  /* ========================================================================== */
  /* DEDICATED FULL-PAGE LEAD SHEET & LYRIC STUDIO VIEW                         */
  /* ========================================================================== */
  #leadsheet-view {
    display: none;
    width: 100%;
    max-width: 1280px;
    padding: 18px 24px 96px;
    flex-direction: column;
    gap: 12px;
  }
  .ls-topbar {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    background: var(--card); border: 1px solid var(--card-border); border-radius: 14px;
    padding: 11px 18px;
  }
  .ls-back-btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: #1e2530; border: 1px solid #313c4d; color: #f8fafc;
    padding: 8px 14px; border-radius: 10px; font-size: 13.5px; font-weight: 700;
    cursor: pointer; transition: 0.15s;
  }
  .ls-back-btn:hover { border-color: var(--accent); color: var(--accent); }
  .ls-title-group { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .ls-title-group h2 { font-size: 17px; font-weight: 700; color: #f8fafc; }

  /* Full-Height Unclipped Chronological Timeline Strip */
  .ls-timeline-card {
    background: #13171d; border: 1px solid var(--card-border); border-radius: 12px;
    padding: 10px 12px; display: flex; flex-direction: column; gap: 6px;
  }
  .ls-timeline-head {
    display: flex; justify-content: space-between; align-items: center; font-size: 11px;
    color: #8b96a5; font-weight: 700; letter-spacing: 0.3px;
  }
  .ls-seq-strip {
    display: flex; align-items: center; gap: 6px; overflow-x: auto;
    padding: 4px 2px 6px; min-height: 54px;
    scrollbar-width: thin; scrollbar-color: #334155 #13171d;
  }
  .ls-seq-item {
    display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
    height: 46px; min-width: 58px; padding: 0 10px; border-radius: 8px;
    background: #1b222d; border: 2px solid #2b3545;
    cursor: pointer; flex-shrink: 0; transition: 0.12s; user-select: none;
  }
  .ls-seq-item:hover { border-color: #94a3b8; filter: brightness(1.12); }
  .ls-seq-item.active {
    border-color: #ffffff; box-shadow: 0 0 14px rgba(249, 115, 22, 0.45); transform: scale(1.04);
  }
  .ls-seq-time { font-size: 10px; color: rgba(255,255,255,0.75); font-variant-numeric: tabular-nums; line-height: 1.1; }
  .ls-seq-chord { font-size: 14px; font-weight: 800; color: #ffffff; line-height: 1.25; }

  /* Brush Palette Bar + Live Current/Next Pills Directly Above Lyric Canvas */
  .ls-palette {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    padding: 9px 14px;
    background: var(--card); border: 1px solid var(--card-border); border-radius: 12px;
  }
  .ls-brush-group {
    display: flex; flex-wrap: wrap; align-items: center; gap: 7px;
  }
  .ls-palette-lbl { font-size: 11.5px; font-weight: 700; color: var(--muted); margin-right: 4px; }
  .ls-drag-chip {
    font-size: 13px; font-weight: 800; padding: 5px 12px; border-radius: 8px;
    background: rgba(249, 115, 22, 0.16); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.45);
    cursor: grab; user-select: none; transition: 0.12s;
  }
  .ls-drag-chip:active { cursor: grabbing; transform: scale(0.96); }
  .ls-drag-chip.selected-brush {
    background: var(--accent); color: #111827; box-shadow: 0 0 14px rgba(249, 115, 22, 0.5);
  }

  /* High-Visibility Current & Next Chord Pills */
  .ls-live-stage {
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }
  .ls-chord-pill {
    display: flex; align-items: center; gap: 9px;
    padding: 5px 13px; border-radius: 10px;
    background: rgba(249, 115, 22, 0.16);
    border: 1.5px solid rgba(249, 115, 22, 0.55);
    box-shadow: 0 0 16px rgba(249, 115, 22, 0.18);
  }
  .ls-chord-pill.next {
    background: #12161d;
    border: 1px solid #2b3545;
    box-shadow: none;
  }
  .ls-pill-tag {
    font-size: 10px; font-weight: 800; letter-spacing: 0.6px;
    text-transform: uppercase; color: #fb923c;
  }
  .ls-chord-pill.next .ls-pill-tag { color: #64748b; }
  .ls-pill-val {
    font-size: 20px; font-weight: 800; color: #ffffff; line-height: 1; min-width: 30px; text-align: center;
  }
  .ls-chord-pill.next .ls-pill-val {
    font-size: 17px; color: #cbd5e1;
  }

  /* Two-Column Full-Height Workspace */
  .ls-workspace {
    display: grid; grid-template-columns: 1fr 1.4fr; gap: 16px; min-height: 440px;
  }
  @media (max-width: 820px) {
    .ls-workspace { grid-template-columns: 1fr; }
  }
  .ls-pane {
    background: var(--card); border: 1px solid var(--card-border); border-radius: 14px;
    padding: 14px 16px; display: flex; flex-direction: column; gap: 8px;
  }
  .ls-pane-lbl { font-size: 11.5px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
  .ls-textarea {
    width: 100%; flex: 1; min-height: 390px; resize: vertical;
    background: #0f1318; border: 1px solid #2b3441; border-radius: 10px;
    padding: 14px 16px; color: #f1f5f9; font-size: 14.5px; line-height: 1.7; outline: none;
  }
  .ls-textarea:focus { border-color: var(--accent); }
  .ls-stage {
    flex: 1; min-height: 390px; max-height: 62vh; overflow-y: auto;
    background: #0f1318; border: 1px solid #2b3441; border-radius: 10px;
    padding: 16px 18px; display: flex; flex-direction: column; gap: 14px;
  }
  .ls-line {
    display: flex; flex-wrap: wrap; align-items: flex-end; gap: 3px; min-height: 48px;
    border-bottom: 1px dashed #1f2733; padding-bottom: 7px;
  }
  .ls-word-slot {
    display: inline-flex; flex-direction: column; align-items: flex-start;
    padding: 2px 5px; border-radius: 6px; cursor: pointer; transition: 0.12s; min-width: 26px;
  }
  .ls-word-slot:hover, .ls-word-slot.drag-over {
    background: rgba(249, 115, 22, 0.16);
  }
  .ls-chord-tag {
    font-size: 12.5px; font-weight: 800; color: #fb923c; background: rgba(249, 115, 22, 0.20);
    border: 1px solid rgba(249, 115, 22, 0.5); border-radius: 5px; padding: 1px 7px;
    margin-bottom: 3px; line-height: 1.25;
  }
  .ls-chord-placeholder {
    height: 21px; width: 100%; border-bottom: 1px dotted transparent; margin-bottom: 3px;
  }
  .ls-word-slot:hover .ls-chord-placeholder {
    border-bottom-color: #64748b;
  }
  .ls-word-text {
    font-size: 15.5px; color: #f8fafc; white-space: pre;
  }

  /* ========================================================================== */
  /* SHARED PERSISTENT BOTTOM PLAYER BAR                                        */
  /* ========================================================================== */
  .bottom-bar {
    position: fixed; bottom: 0; left: 0; right: 0; background: #12161c;
    border-top: 1px solid var(--card-border); padding: 12px 22px;
    display: flex; justify-content: center; z-index: 50;
  }
  .bottom-inner {
    width: 100%; max-width: 1180px; display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap;
  }
  .player-left { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 240px; }
  .play-btn {
    width: 44px; height: 44px; border-radius: 50%; border: none; background: var(--accent);
    color: #111; font-size: 17px; cursor: pointer; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: transform 0.1s;
  }
  .play-btn:hover { transform: scale(1.05); }
  .skip-btn {
    display: none; align-items: center; justify-content: center;
    height: 36px; padding: 0 11px; border-radius: 9px;
    background: #1b222c; border: 1px solid #2d3748; color: #e2e8f0;
    font-size: 12.5px; font-weight: 700; cursor: pointer; transition: 0.15s; white-space: nowrap;
  }
  .skip-btn:hover { border-color: var(--accent); color: #fff; }
  .time-display { font-size: 13px; font-variant-numeric: tabular-nums; color: #cbd5e1; min-width: 86px; }
  .seek-bar { flex: 1; accent-color: var(--accent); cursor: pointer; height: 5px; }

  .controls-right { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
  .capo-badge {
    display: none; font-size: 11.5px; font-weight: 700; padding: 6px 9px; border-radius: 8px;
    background: rgba(249, 115, 22, 0.14); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.35);
    white-space: nowrap;
  }
  .transpose-label { font-size: 13px; color: var(--muted); margin-right: 2px; }
  .btn-ctrl {
    background: #1b2129; border: 1px solid var(--card-border); color: var(--text);
    height: 36px; padding: 0 11px; border-radius: 9px; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: 0.15s; white-space: nowrap;
  }
  .btn-ctrl:hover { background: #252d38; border-color: #3b4758; }
  .btn-midi {
    background: rgba(56, 189, 248, 0.13); border-color: rgba(56, 189, 248, 0.38); color: #38bdf8;
  }
  .btn-midi:hover { background: rgba(56, 189, 248, 0.22); }
  .btn-lyrics {
    background: rgba(167, 139, 250, 0.14); border-color: rgba(167, 139, 250, 0.4); color: #c084fc;
  }
  .btn-lyrics:hover { background: rgba(167, 139, 250, 0.24); }
  .btn-pdf {
    background: rgba(249, 115, 22, 0.16); border-color: rgba(249, 115, 22, 0.45); color: #fb923c;
  }
  .btn-pdf:hover { background: rgba(249, 115, 22, 0.26); }
  .transpose-val {
    background: #1b2129; border: 1px solid var(--card-border); height: 36px; min-width: 44px;
    border-radius: 9px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;
  }

  /* Modern Glassmorphism Center Modals (Confirm & Edit) */
  @keyframes modalPopIn {
    from { opacity: 0; transform: scale(0.95) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  .modal-backdrop {
    display: none; position: fixed; inset: 0; background: rgba(9, 12, 17, 0.80);
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    z-index: 120; align-items: center; justify-content: center; padding: 20px;
  }
  .modal-backdrop.open { display: flex; }
  .modern-modal-card {
    width: 100%; max-width: 430px; background: linear-gradient(180deg, #191f28 0%, #131820 100%);
    border: 1px solid #2e3847; border-radius: 18px; padding: 24px 26px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255,255,255,0.03) inset;
    animation: modalPopIn 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .modal-icon-badge {
    width: 42px; height: 42px; border-radius: 12px;
    background: rgba(192, 132, 252, 0.14); border: 1px solid rgba(192, 132, 252, 0.35);
    color: #c084fc; display: flex; align-items: center; justify-content: center;
    font-size: 19px; margin-bottom: 14px;
  }
  .modal-title { font-size: 17px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }
  .modal-desc { font-size: 13.5px; color: #94a3b8; line-height: 1.5; margin-bottom: 16px; }

  .sig-switch-preview {
    display: flex; align-items: center; justify-content: center; gap: 14px;
    background: #0f1319; border: 1px solid #252e3b; border-radius: 12px;
    padding: 12px; margin-bottom: 18px;
  }
  .sig-chip-old {
    font-size: 15px; font-weight: 800; padding: 5px 12px; border-radius: 8px;
    background: #1d2430; color: #94a3b8; border: 1px solid #2e3847;
  }
  .sig-arrow { color: var(--accent); font-size: 16px; font-weight: 800; }
  .sig-chip-new {
    font-size: 15px; font-weight: 800; padding: 5px 12px; border-radius: 8px;
    background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.45);
  }

  .modal-btn-row { display: flex; gap: 10px; justify-content: flex-end; }
  .btn-modal-ghost {
    flex: 1; background: #1d2430; color: #cbd5e1; border: 1px solid #2e3847;
    border-radius: 10px; padding: 10px 14px; font-weight: 600; font-size: 13.5px; cursor: pointer; transition: 0.15s;
  }
  .btn-modal-ghost:hover { background: #262f3e; color: #fff; }
  .btn-modal-accent {
    flex: 1.3; background: var(--accent); color: #111827; border: none;
    border-radius: 10px; padding: 10px 16px; font-weight: 700; font-size: 13.5px; cursor: pointer; transition: 0.15s;
  }
  .btn-modal-accent:hover { filter: brightness(1.08); }

  .edit-select {
    width: 100%; background: #0f1318; border: 1px solid #2e3846; color: #f8fafc;
    border-radius: 10px; padding: 10px 12px; font-size: 14px; font-weight: 600; margin-bottom: 16px; outline: none;
  }
  .edit-select:focus { border-color: var(--accent); }
  .edit-actions { display: flex; flex-direction: column; gap: 8px; }

  /* Toast Notification */
  .toast-box {
    position: fixed; bottom: 84px; right: 24px; z-index: 150;
    background: #19202a; border: 1px solid #2f3b4c; color: #f1f5f9;
    padding: 11px 16px; border-radius: 11px; font-size: 13px; font-weight: 600;
    box-shadow: 0 12px 30px rgba(0,0,0,0.5); display: none; align-items: center; gap: 8px;
    animation: modalPopIn 0.18s ease;
  }
  .toast-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }

  /* Progress Loader */
  .loader {
    display: none; position: fixed; inset: 0; background: rgba(11, 15, 20, 0.88);
    backdrop-filter: blur(6px); z-index: 100; align-items: center; justify-content: center; padding: 20px;
  }
  .progress-card {
    width: 100%; max-width: 520px; background: #161b22; border: 1px solid #2b3441;
    border-radius: 16px; padding: 26px 28px; box-shadow: 0 20px 50px rgba(0,0,0,0.55);
  }
  .prog-top {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
  }
  .prog-stage { font-size: 16px; font-weight: 700; color: #f8fafc; }
  .prog-pct {
    font-size: 22px; font-weight: 800; color: var(--accent); font-variant-numeric: tabular-nums;
  }
  .prog-bar-bg {
    width: 100%; height: 11px; background: #222933; border-radius: 999px; overflow: hidden;
    margin-bottom: 16px; border: 1px solid #2e3744;
  }
  .prog-bar-fill {
    height: 100%; width: 0%; background: linear-gradient(90deg, #f97316, #fb923c);
    border-radius: 999px; transition: width 0.25s ease;
  }
  .prog-stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px;
  }
  .stat-box {
    background: #11151b; border: 1px solid #232b36; border-radius: 10px; padding: 10px 12px;
  }
  .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px; }
  .stat-value { font-size: 14.5px; font-weight: 700; color: #e2e8f0; font-variant-numeric: tabular-nums; }
  .prog-steps {
    display: flex; justify-content: space-between; gap: 6px; margin-bottom: 14px;
  }
  .step-pill {
    flex: 1; text-align: center; font-size: 11px; font-weight: 700; padding: 5px 4px;
    border-radius: 6px; background: #1c222b; color: #64748b; border: 1px solid #262f3b;
  }
  .step-pill.active { background: rgba(249, 115, 22, 0.16); color: #fb923c; border-color: rgba(249, 115, 22, 0.45); }
  .step-pill.done { background: rgba(34, 197, 94, 0.14); color: #4ade80; border-color: rgba(34, 197, 94, 0.35); }
  .prog-detail {
    font-size: 12.5px; color: #94a3b8; background: #101419; border: 1px solid #202731;
    padding: 9px 12px; border-radius: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
</style>
</head>
<body>

<!-- Toast Notification -->
<div class="toast-box" id="toast-box">
  <span class="toast-dot"></span>
  <span id="toast-msg"></span>
</div>

<!-- Live Progress Modal -->
<div class="loader" id="loader">
  <div class="progress-card">
    <div class="prog-top">
      <div class="prog-stage" id="prog-stage">Preparing...</div>
      <div class="prog-pct" id="prog-pct">0%</div>
    </div>
    <div class="prog-bar-bg">
      <div class="prog-bar-fill" id="prog-bar-fill"></div>
    </div>
    <div class="prog-stats">
      <div class="stat-box">
        <div class="stat-label">Elapsed</div>
        <div class="stat-value" id="stat-elapsed">0:00</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Time Left</div>
        <div class="stat-value" id="stat-eta">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label" id="stat-speed-lbl">Speed</div>
        <div class="stat-value" id="stat-speed">--</div>
      </div>
    </div>
    <div class="prog-steps">
      <div class="step-pill" id="step-1">1. Load</div>
      <div class="step-pill" id="step-2">2. AI Stems</div>
      <div class="step-pill" id="step-3">3. Beat Sync</div>
      <div class="step-pill" id="step-4">4. Chords</div>
    </div>
    <div class="prog-detail" id="prog-detail">Initializing audio pipeline...</div>
  </div>
</div>

<!-- Modern Time Signature Confirmation Modal -->
<div class="modal-backdrop" id="sig-modal">
  <div class="modern-modal-card">
    <div class="modal-icon-badge">&#9834;</div>
    <div class="modal-title">Change Time Signature?</div>
    <div class="modal-desc">
      Re-align detected chord switches and downbeat grid to the new meter:
    </div>
    <div class="sig-switch-preview">
      <span class="sig-chip-old" id="sig-old-val">4/4</span>
      <span class="sig-arrow">&rarr;</span>
      <span class="sig-chip-new" id="sig-new-val">3/4</span>
    </div>
    <div class="modal-btn-row">
      <button type="button" class="btn-modal-ghost" id="btn-sig-cancel">Cancel</button>
      <button type="button" class="btn-modal-accent" id="btn-sig-confirm">Re-Align Chords</button>
    </div>
  </div>
</div>

<!-- Chord Edit / Global Replace Modal -->
<div class="modal-backdrop" id="edit-modal">
  <div class="modern-modal-card">
    <div class="modal-title" id="edit-title">Edit Chord</div>
    <div class="modal-desc" id="edit-sub">Replace this chord block or fix all matching chords across the song:</div>
    <select class="edit-select" id="edit-select"></select>
    <div class="edit-actions">
      <button type="button" class="btn-modal-accent" id="btn-replace-all">Replace All in Entire Song</button>
      <button type="button" class="btn-modal-ghost" id="btn-replace-one">Change Only This Block</button>
      <button type="button" class="btn-modal-ghost" id="btn-edit-cancel" style="border-color:transparent;color:#64748b;">Cancel</button>
    </div>
  </div>
</div>

<!-- ======================================================================== -->
<!-- VIEW 1: MAIN CHORD DETECTOR VIEW                                         -->
<!-- ======================================================================== -->
<div class="container" id="detector-view">
  <header>
    <div class="brand">
      <div class="logo-box">&#9835;</div>
      <h1>Chord Detector</h1>
      <span class="badge">PRO STUDIO</span>
      <div class="meta-group-top" id="meta-group-top">
        <span class="meta-pill" id="song-meta"></span>
        <button type="button" class="sig-pill" id="sig-pill" title="Click to change Time Signature & re-align AI chords">4/4</button>
        <button type="button" class="bpm-mult-btn" id="bpm-half" title="Half-time BPM (&divide;2)">&frac12;x</button>
        <button type="button" class="bpm-mult-btn" id="bpm-double" title="Double-time BPM (&times;2)">2x</button>
      </div>
    </div>
    <label class="toggle-wrap">
      <span class="switch">
        <input type="checkbox" id="demucs-toggle" />
        <span class="slider"></span>
      </span>
      <span>Demucs stems</span>
    </label>
  </header>

  <div class="input-grid">
    <div class="dropzone" id="dropzone">
      <div class="upload-icon">&#8679;</div>
      <div>
        <div class="drop-title">Drop a song file here</div>
        <div class="drop-sub">or click to browse &middot; MP3, WAV, M4A, FLAC</div>
      </div>
      <input type="file" id="file-input" accept=".mp3,.wav,.m4a,.flac,audio/*" style="display:none" />
    </div>

    <div class="url-box">
      <div class="url-label">&#128279; Paste YouTube or Direct Audio Link</div>
      <div class="url-row">
        <input type="text" class="url-input" id="url-input" placeholder="https://www.youtube.com/watch?v=..." />
        <button type="button" class="url-btn" id="url-btn">Analyze Link</button>
      </div>
    </div>
  </div>

  <div class="stage-card">
    <div class="song-title" id="song-title">Upload an audio track or paste a YouTube link to begin</div>
    <div class="stage-body">
      <div class="stage-center">
        <div class="chord-short" id="chord-short">&mdash;</div>
        <div class="chord-full" id="chord-full">Ready</div>
        <div class="chord-next">Next <strong id="chord-next">&mdash;</strong></div>
      </div>

      <div class="diagram-panel">
        <div class="inst-tabs">
          <button type="button" class="inst-tab active" id="tab-guitar">Guitar</button>
          <button type="button" class="inst-tab" id="tab-piano">Piano</button>
        </div>
        <div id="diagram-container"></div>
      </div>

      <div class="circle-panel">
        <div class="circle-header">
          <span>CIRCLE OF FIFTHS</span>
          <span class="circle-key-tag" id="circle-key-tag">C Major</span>
        </div>
        <div id="circle-container"></div>
        <div class="scale-chips" id="scale-chips"></div>
      </div>
    </div>
  </div>

  <div class="timeline-wrapper" id="timeline-wrapper">
    <div class="timeline-track" id="timeline-track"></div>
  </div>

  <div class="shortcut-hint">
    Click <strong>4/4 &middot; 3/4 &middot; 6/8</strong> pill to re-align meter &nbsp;&middot;&nbsp;
    <strong>Double-click</strong> block to edit chord &nbsp;&middot;&nbsp;
    Export <strong>.mid</strong> for FL Studio or open <strong>Lead Sheet</strong>
  </div>
</div>

<!-- ======================================================================== -->
<!-- VIEW 2: DEDICATED FULL-PAGE LEAD SHEET & LYRIC OVERLAY STUDIO            -->
<!-- ======================================================================== -->
<div id="leadsheet-view">
  <div class="ls-topbar">
    <div class="ls-title-group">
      <button type="button" class="ls-back-btn" id="ls-back-btn">&larr; Back to Detector</button>
      <h2>Lead Sheet Studio</h2>
      <span class="meta-pill" id="ls-song-badge">No Track Loaded</span>
    </div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button type="button" class="btn-ctrl" id="ls-clear-chords">Clear Placed Chords</button>
      <button type="button" class="btn-ctrl btn-pdf" id="ls-print-btn">Print / Save Lead Sheet PDF</button>
    </div>
  </div>

  <!-- Full-Height Scrollable Song Chord Timeline -->
  <div class="ls-timeline-card">
    <div class="ls-timeline-head">
      <span>SONG CHORD TIMELINE (Click any block to jump audio &amp; select chord)</span>
      <span id="ls-timeline-count">0 segments</span>
    </div>
    <div class="ls-seq-strip" id="ls-seq-strip"></div>
  </div>

  <!-- Quick Chord Brush Palette + Live Current & Next Chord Pills Right Above Lyrics -->
  <div class="ls-palette" id="ls-palette">
    <div class="ls-brush-group" id="ls-brush-group">
      <span class="ls-palette-lbl">CHORD BRUSH (Drag or Click-Select, then tap words below):</span>
    </div>
    <div class="ls-live-stage">
      <div class="ls-chord-pill" title="Currently playing chord">
        <span class="ls-pill-tag">Current</span>
        <span class="ls-pill-val" id="ls-curr-chord">&mdash;</span>
      </div>
      <div class="ls-chord-pill next" title="Upcoming chord">
        <span class="ls-pill-tag">Next</span>
        <span class="ls-pill-val" id="ls-next-chord">&mdash;</span>
      </div>
    </div>
  </div>

  <!-- Full-Width Two-Column Lyrics & Placement Canvas -->
  <div class="ls-workspace">
    <div class="ls-pane">
      <div class="ls-pane-lbl">1. Paste or Edit Song Lyrics</div>
      <textarea class="ls-textarea" id="ls-lyrics-input" placeholder="Paste your song lyrics here...&#10;&#10;Example:&#10;Uski mahima karo, bandhan toot jayenge&#10;Mera Yeshu maujood hai yahan..."></textarea>
    </div>
    <div class="ls-pane">
      <div class="ls-pane-lbl">2. Click or Drop Chords Above Exact Syllables</div>
      <div class="ls-stage" id="ls-stage"></div>
    </div>
  </div>
</div>

<!-- ======================================================================== -->
<!-- PERSISTENT BOTTOM PLAYER BAR (ADAPTS MINIMALISTICALLY ON LEAD SHEET PAGE) -->
<!-- ======================================================================== -->
<div class="bottom-bar">
  <div class="bottom-inner">
    <div class="player-left">
      <button type="button" class="skip-btn" id="skip-back-btn" title="Rewind 5 seconds">&minus;5s</button>
      <button type="button" class="play-btn" id="play-btn">&#9654;</button>
      <button type="button" class="skip-btn" id="skip-fwd-btn" title="Forward 5 seconds">+5s</button>
      <div class="time-display" id="time-display">0:00 / 0:00</div>
      <input type="range" class="seek-bar" id="seek-bar" min="0" max="100" step="0.05" value="0" />
    </div>

    <div class="controls-right" id="detector-bottom-controls">
      <span class="capo-badge" id="capo-badge"></span>
      <span class="transpose-label">Transpose</span>
      <button type="button" class="btn-ctrl" id="trans-down">&minus;</button>
      <div class="transpose-val" id="trans-val">+0</div>
      <button type="button" class="btn-ctrl" id="trans-up">+</button>
      <button type="button" class="btn-ctrl btn-midi" id="dl-midi" title="Export multi-note MIDI chord track for FL Studio / DAW">.mid</button>
      <button type="button" class="btn-ctrl btn-lyrics" id="open-leadsheet" title="Open full-page Lead Sheet & Lyric Overlay Studio">Lead Sheet</button>
      <button type="button" class="btn-ctrl" id="dl-txt">.txt</button>
      <button type="button" class="btn-ctrl" id="dl-csv">.csv</button>
      <button type="button" class="btn-ctrl btn-pdf" id="dl-pdf">PDF / Print</button>
    </div>

    <div class="controls-right" id="leadsheet-bottom-controls" style="display:none;">
      <button type="button" class="btn-ctrl btn-pdf" id="ls-bar-print-btn">Print / Save Lead Sheet PDF</button>
    </div>
  </div>
</div>

<audio id="audio-player" style="display:none"></audio>

<script>
  const PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const CIRCLE_MAJOR = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"];
  const CIRCLE_MINOR = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m", "Fm", "Cm", "Gm", "Dm"];
  const TIME_SIGS = ["4/4", "3/4", "6/8"];

  const ROOT_COLORS = {
    "C": "#6e692c", "C#": "#2d6a6a", "D": "#7d792d", "D#": "#3d5a80",
    "E": "#7a4458", "F": "#2f6f4e", "F#": "#5b4b8a", "G": "#2e6b62",
    "G#": "#7c4d2b", "A": "#6b4c7a", "A#": "#8f2d56", "B": "#386641"
  };

  const GUITAR_VOICINGS = {
    "C Major":  [-1, 3, 2, 0, 1, 0], "C Minor":  [-1, 3, 5, 5, 4, 3],
    "C# Major": [-1, 4, 6, 6, 6, 4], "C# Minor": [-1, 4, 6, 6, 5, 4],
    "D Major":  [-1, -1, 0, 2, 3, 2],"D Minor":  [-1, -1, 0, 2, 3, 1],
    "D# Major": [-1, 6, 8, 8, 8, 6], "D# Minor": [-1, 6, 8, 8, 7, 6],
    "E Major":  [0, 2, 2, 1, 0, 0],  "E Minor":  [0, 2, 2, 0, 0, 0],
    "F Major":  [1, 3, 3, 2, 1, 1],  "F Minor":  [1, 3, 3, 1, 1, 1],
    "F# Major": [2, 4, 4, 3, 2, 2],  "F# Minor": [2, 4, 4, 2, 2, 2],
    "G Major":  [3, 2, 0, 0, 0, 3],  "G Minor":  [3, 5, 5, 3, 3, 3],
    "G# Major": [4, 6, 6, 5, 4, 4],  "G# Minor": [4, 6, 6, 4, 4, 4],
    "A Major":  [-1, 0, 2, 2, 2, 0], "A Minor":  [-1, 0, 2, 2, 1, 0],
    "A# Major": [-1, 1, 3, 3, 3, 1], "A# Minor": [-1, 1, 3, 3, 2, 1],
    "B Major":  [-1, 2, 4, 4, 4, 2], "B Minor":  [-1, 2, 4, 4, 3, 2]
  };

  let rawChords = [];
  let songKey = "C Major";
  let songBpm = 0;
  let songTimeSig = "4/4";
  let pendingTimeSig = "4/4";
  let songName = "detected_chords";
  let transposeSteps = 0;
  let activeIndex = -1;
  let diagramMode = "guitar";
  let currentFullChord = "N";
  let rafId = null;
  let editingBlockIdx = -1;
  let currentAudioId = "";
  let toastTimer = null;
  let currentRoute = "detector";

  let leadSheetPlacements = {};
  let selectedBrushChord = "";

  const audio = document.getElementById("audio-player");
  const playBtn = document.getElementById("play-btn");
  const skipBackBtn = document.getElementById("skip-back-btn");
  const skipFwdBtn = document.getElementById("skip-fwd-btn");
  const seekBar = document.getElementById("seek-bar");
  const timeDisplay = document.getElementById("time-display");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  const detectorView = document.getElementById("detector-view");
  const leadSheetView = document.getElementById("leadsheet-view");
  const detectorBottomControls = document.getElementById("detector-bottom-controls");
  const leadSheetBottomControls = document.getElementById("leadsheet-bottom-controls");
  const lyricsInput = document.getElementById("ls-lyrics-input");
  const lsStage = document.getElementById("ls-stage");
  const lsBrushGroup = document.getElementById("ls-brush-group");
  const lsSeqStrip = document.getElementById("ls-seq-strip");

  function showToast(msg) {
    const box = document.getElementById("toast-box");
    document.getElementById("toast-msg").textContent = msg;
    box.style.display = "flex";
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.style.display = "none"; }, 3200);
  }

  // ============================================================================
  // PAGE VIEW ROUTING (MAIN DETECTOR <-> MINIMALISTIC LEAD SHEET STUDIO)
  // ============================================================================
  function navigateToView(viewName, pushHistory = true) {
    currentRoute = viewName;
    if (viewName === "leadsheet") {
      detectorView.style.display = "none";
      leadSheetView.style.display = "flex";
      detectorBottomControls.style.display = "none";
      leadSheetBottomControls.style.display = "flex";
      skipBackBtn.style.display = "inline-flex";
      skipFwdBtn.style.display = "inline-flex";

      const tKey = transposeLabel(songKey, transposeSteps);
      document.getElementById("ls-song-badge").textContent =
        rawChords.length ? `${songName} • ${tKey} • ${songTimeSig}` : "Ready (Paste Lyrics)";
      renderLeadSheetSequenceStrip();
      renderLeadSheetPalette();
      renderLeadSheetStage();
      syncPlaybackUI(true);
      window.scrollTo({ top: 0, behavior: "instant" });
      if (pushHistory && window.location.hash !== "#leadsheet") {
        history.pushState({ view: "leadsheet" }, "", "#leadsheet");
      }
    } else {
      leadSheetView.style.display = "none";
      detectorView.style.display = "block";
      detectorBottomControls.style.display = "flex";
      leadSheetBottomControls.style.display = "none";
      skipBackBtn.style.display = "none";
      skipFwdBtn.style.display = "none";

      syncPlaybackUI(true);
      if (pushHistory && window.location.hash === "#leadsheet") {
        history.pushState({ view: "detector" }, "", window.location.pathname + window.location.search);
      }
    }
  }

  document.getElementById("open-leadsheet").onclick = () => navigateToView("leadsheet", true);
  document.getElementById("ls-back-btn").onclick = () => {
    if (window.location.hash === "#leadsheet") {
      history.back();
    } else {
      navigateToView("detector", false);
    }
  };
  window.addEventListener("popstate", () => {
    if (window.location.hash === "#leadsheet") {
      navigateToView("leadsheet", false);
    } else {
      navigateToView("detector", false);
    }
  });

  (function initEditDropdown() {
    const sel = document.getElementById("edit-select");
    PITCHES.forEach(p => {
      ["Major", "Minor"].forEach(q => {
        const full = `${p} ${q}`;
        const opt = document.createElement("option");
        opt.value = full;
        opt.textContent = `${toShortChord(full)} (${full})`;
        sel.appendChild(opt);
      });
    });
  })();

  function fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? "0" : ""}${s}`;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function resetAudioSource(newSrc) {
    audio.pause();
    audio.currentTime = 0;
    playBtn.innerHTML = "&#9654;";
    seekBar.value = 0;
    timeDisplay.textContent = "0:00 / 0:00";
    audio.src = newSrc;
    audio.load();
  }

  function setStepState(activeStep, useDemucs) {
    for (let i = 1; i <= 4; i++) {
      const el = document.getElementById(`step-${i}`);
      el.classList.remove("active", "done");
      if (i === 2 && !useDemucs) {
        el.textContent = "2. Fast Mode";
      } else if (i === 2) {
        el.textContent = "2. AI Stems";
      }
      if (i < activeStep) el.classList.add("done");
      else if (i === activeStep) el.classList.add("active");
    }
  }

  function updateProgressModal({ percent, stage, detail, eta, speed, speedLbl, step, useDemucs }) {
    if (percent !== undefined) {
      document.getElementById("prog-pct").textContent = `${percent}%`;
      document.getElementById("prog-bar-fill").style.width = `${percent}%`;
    }
    if (stage !== undefined) document.getElementById("prog-stage").textContent = stage;
    if (detail !== undefined) document.getElementById("prog-detail").textContent = detail;
    if (eta !== undefined) document.getElementById("stat-eta").textContent = eta;
    if (speed !== undefined) document.getElementById("stat-speed").textContent = speed;
    if (speedLbl !== undefined) document.getElementById("stat-speed-lbl").textContent = speedLbl;
    if (step !== undefined) setStepState(step, useDemucs);
  }

  function transposeLabel(label, steps) {
    if (!label || label === "N") return label;
    const parts = label.split(" ");
    const root = parts[0];
    const qual = parts.slice(1).join(" ");
    const idx = PITCHES.indexOf(root);
    if (idx === -1) return label;
    const newRoot = PITCHES[(idx + steps + 120) % 12];
    return qual ? `${newRoot} ${qual}` : newRoot;
  }

  function toShortChord(label) {
    if (!label || label === "N") return "—";
    const [root, qual] = label.split(" ");
    if (qual === "Major") return root;
    if (qual === "Minor") return `${root}m`;
    return label;
  }

  function shortToFullChord(shortStr) {
    if (!shortStr) return "N";
    if (shortStr.endsWith("m")) {
      return `${shortStr.slice(0, -1)} Minor`;
    }
    return `${shortStr} Major`;
  }

  function getRootColor(label) {
    if (!label || label === "N") return "#232a34";
    const root = label.split(" ")[0];
    return ROOT_COLORS[root] || "#4b5563";
  }

  // ============================================================================
  // PURE-JS TYPE-0 MULTI-NOTE MIDI GENERATOR (FOR FL STUDIO / ANY DAW)
  // ============================================================================
  function chordLabelToMidiNotes(label) {
    if (!label || label === "N") return [];
    const [root, qual] = label.split(" ");
    const rIdx = PITCHES.indexOf(root);
    if (rIdx === -1) return [];
    const bass = 48 + rIdx;
    const root4 = 60 + rIdx;
    const third = root4 + (qual === "Minor" ? 3 : 4);
    const fifth = root4 + 7;
    return [bass, root4, third, fifth];
  }

  function encodeVarLen(value) {
    let buffer = value & 0x7F;
    const bytes = [];
    while ((value >>= 7)) {
      buffer <<= 8;
      buffer |= ((value & 0x7F) | 0x80);
    }
    while (true) {
      bytes.push(buffer & 0xFF);
      if (buffer & 0x80) buffer >>= 8;
      else break;
    }
    return bytes;
  }

  function exportChordsToMidi() {
    if (!rawChords.length) return showToast("Analyze a song first to export MIDI");

    const bpm = songBpm > 0 ? songBpm : 100;
    const ticksPerQuarter = 480;
    const microsPerQuarter = Math.round(60000000 / bpm);
    const ticksPerSecond = (ticksPerQuarter * bpm) / 60;

    const [numStr, denStr] = songTimeSig.split("/");
    const numerator = parseInt(numStr, 10) || 4;
    const denominatorPower = Math.round(Math.log2(parseInt(denStr, 10) || 4));

    const trackBytes = [];

    trackBytes.push(
      0x00, 0xFF, 0x51, 0x03,
      (microsPerQuarter >> 16) & 0xFF,
      (microsPerQuarter >> 8) & 0xFF,
      microsPerQuarter & 0xFF
    );

    trackBytes.push(
      0x00, 0xFF, 0x58, 0x04,
      numerator & 0xFF,
      denominatorPower & 0xFF,
      0x18, 0x08
    );

    const events = [];
    rawChords.forEach(seg => {
      const transposed = transposeLabel(seg.chord, transposeSteps);
      const notes = chordLabelToMidiNotes(transposed);
      if (!notes.length) return;

      const startTick = Math.max(0, Math.round(seg.start * ticksPerSecond));
      const endTick = Math.max(startTick + 60, Math.round(seg.end * ticksPerSecond) - 10);

      notes.forEach(n => {
        events.push({ tick: startTick, type: 0x90, note: n, vel: 92 });
        events.push({ tick: endTick, type: 0x80, note: n, vel: 0 });
      });
    });

    events.sort((a, b) => (a.tick - b.tick) || (a.type - b.type));

    let lastTick = 0;
    events.forEach(ev => {
      const delta = Math.max(0, ev.tick - lastTick);
      lastTick = ev.tick;
      trackBytes.push(...encodeVarLen(delta), ev.type, ev.note & 0x7F, ev.vel & 0x7F);
    });

    trackBytes.push(0x00, 0xFF, 0x2F, 0x00);

    const header = [
      0x4D, 0x54, 0x68, 0x64,
      0x00, 0x00, 0x00, 0x06,
      0x00, 0x00,
      0x00, 0x01,
      (ticksPerQuarter >> 8) & 0xFF,
      ticksPerQuarter & 0xFF
    ];

    const trkLen = trackBytes.length;
    const trackHeader = [
      0x4D, 0x54, 0x72, 0x6B,
      (trkLen >> 24) & 0xFF,
      (trkLen >> 16) & 0xFF,
      (trkLen >> 8) & 0xFF,
      trkLen & 0xFF
    ];

    const midiBinary = new Uint8Array([...header, ...trackHeader, ...trackBytes]);
    const blob = new Blob([midiBinary], { type: "audio/midi" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${songName}_chords.mid`;
    a.click();
    showToast(`Exported ${songName}_chords.mid (${bpm} BPM, ${songTimeSig})`);
  }

  document.getElementById("dl-midi").onclick = exportChordsToMidi;

  // ============================================================================
  // FULL-PAGE LEAD SHEET & LYRIC OVERLAY STUDIO
  // ============================================================================
  function getUniqueSongChords() {
    const seen = new Set();
    const list = [];
    rawChords.forEach(item => {
      const shortC = toShortChord(transposeLabel(item.chord, transposeSteps));
      if (shortC && shortC !== "—" && !seen.has(shortC)) {
        seen.add(shortC);
        list.push(shortC);
      }
    });
    if (!list.length) {
      return ["C", "G", "Am", "F", "D", "Em"];
    }
    return list;
  }

  function renderLeadSheetSequenceStrip() {
    document.getElementById("ls-timeline-count").textContent = `${rawChords.length} segments`;
    if (!rawChords.length) {
      lsSeqStrip.innerHTML = `<span style="font-size:12.5px;color:#64748b;padding:4px 8px;">Analyze a song first to see its full timestamped chord progression here</span>`;
      return;
    }
    lsSeqStrip.innerHTML = "";
    rawChords.forEach((item, idx) => {
      const fullLabel = transposeLabel(item.chord, transposeSteps);
      const shortC = toShortChord(fullLabel);
      const div = document.createElement("div");
      div.className = "ls-seq-item" + (idx === activeIndex ? " active" : "");
      div.dataset.idx = idx;
      div.style.background = getRootColor(fullLabel);
      div.innerHTML = `<span class="ls-seq-time">${fmtTime(item.start)}</span><span class="ls-seq-chord">${escapeHtml(shortC)}</span>`;
      div.onclick = () => {
        audio.currentTime = item.start + 0.005;
        if (audio.paused) audio.play();
        if (shortC && shortC !== "—") {
          selectedBrushChord = shortC;
          renderLeadSheetPalette();
        }
        syncPlaybackUI(true);
      };
      lsSeqStrip.appendChild(div);
    });
  }

  function renderLeadSheetPalette() {
    const chords = getUniqueSongChords();
    if (!selectedBrushChord || !chords.includes(selectedBrushChord)) {
      selectedBrushChord = chords[0];
    }
    lsBrushGroup.innerHTML = `<span class="ls-palette-lbl">CHORD BRUSH (Drag or Click-Select, then tap words below):</span>` +
      chords.map(c => `
        <span class="ls-drag-chip ${c === selectedBrushChord ? 'selected-brush' : ''}"
              draggable="true"
              data-chord="${escapeHtml(c)}">
          ${escapeHtml(c)}
        </span>
      `).join("");

    lsBrushGroup.querySelectorAll(".ls-drag-chip").forEach(chip => {
      chip.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", chip.dataset.chord);
        selectedBrushChord = chip.dataset.chord;
        renderLeadSheetPalette();
      });
      chip.addEventListener("click", () => {
        selectedBrushChord = chip.dataset.chord;
        renderLeadSheetPalette();
      });
    });
  }

  function renderLeadSheetStage() {
    const text = lyricsInput.value;
    if (!text.trim()) {
      lsStage.innerHTML = `<div style="color:#64748b;font-size:14px;text-align:center;margin:auto;padding:24px;">
        Paste or type your lyrics in the left panel to start placing chords above words.
      </div>`;
      return;
    }

    const lines = text.split(/\r?\n/);
    lsStage.innerHTML = "";

    lines.forEach((line, lineIdx) => {
      const lineDiv = document.createElement("div");
      lineDiv.className = "ls-line";

      if (!line.trim()) {
        lineDiv.style.minHeight = "22px";
        lineDiv.style.borderBottom = "none";
        lsStage.appendChild(lineDiv);
        return;
      }

      const words = line.split(/(\s+)/).filter(tok => tok.trim().length > 0);
      words.forEach((word, wordIdx) => {
        const key = `${lineIdx}:${wordIdx}`;
        const assignedChord = leadSheetPlacements[key];

        const slot = document.createElement("div");
        slot.className = "ls-word-slot";
        slot.title = assignedChord
          ? `Click to remove ${assignedChord} (or drop new chord)`
          : `Click or drop "${selectedBrushChord}" above "${word}"`;

        if (assignedChord) {
          slot.innerHTML = `
            <span class="ls-chord-tag">${escapeHtml(assignedChord)}</span>
            <span class="ls-word-text">${escapeHtml(word)} </span>
          `;
        } else {
          slot.innerHTML = `
            <span class="ls-chord-placeholder"></span>
            <span class="ls-word-text">${escapeHtml(word)} </span>
          `;
        }

        slot.onclick = () => {
          if (leadSheetPlacements[key] === selectedBrushChord) {
            delete leadSheetPlacements[key];
          } else if (selectedBrushChord) {
            leadSheetPlacements[key] = selectedBrushChord;
          }
          renderLeadSheetStage();
        };

        slot.ondragover = (e) => {
          e.preventDefault();
          slot.classList.add("drag-over");
        };
        slot.ondragleave = () => slot.classList.remove("drag-over");
        slot.ondrop = (e) => {
          e.preventDefault();
          slot.classList.remove("drag-over");
          const droppedChord = e.dataTransfer.getData("text/plain") || selectedBrushChord;
          if (droppedChord) {
            leadSheetPlacements[key] = droppedChord;
            renderLeadSheetStage();
          }
        };

        lineDiv.appendChild(slot);
      });

      lsStage.appendChild(lineDiv);
    });
  }

  lyricsInput.addEventListener("input", renderLeadSheetStage);
  document.getElementById("ls-clear-chords").onclick = () => {
    leadSheetPlacements = {};
    renderLeadSheetStage();
    showToast("Cleared all placed lyric chords");
  };

  function printLeadSheet() {
    const text = lyricsInput.value.trim();
    if (!text) return showToast("Paste some lyrics first");

    const tKey = transposeLabel(songKey, transposeSteps);
    let capoBadge = "";
    if (transposeSteps < 0) {
      capoBadge = `<span class="pill capo">Capo ${ordinal(Math.abs(transposeSteps))} Fret</span>`;
    } else if (transposeSteps > 0) {
      capoBadge = `<span class="pill capo">Transpose +${transposeSteps}</span>`;
    }

    const lines = lyricsInput.value.split(/\r?\n/);
    const renderedLines = lines.map((line, lineIdx) => {
      if (!line.trim()) return `<div class="spacer"></div>`;
      const words = line.split(/(\s+)/).filter(tok => tok.trim().length > 0);
      const tokensHtml = words.map((word, wordIdx) => {
        const ch = leadSheetPlacements[`${lineIdx}:${wordIdx}`] || "&nbsp;";
        return `<span class="tok"><span class="ch">${escapeHtml(ch).replace("&amp;nbsp;", "&nbsp;")}</span><span class="wd">${escapeHtml(word)}&nbsp;</span></span>`;
      }).join("");
      return `<div class="lyric-line">${tokensHtml}</div>`;
    }).join("");

    const printWin = window.open("", "_blank", "width=920,height=780");
    if (!printWin) return showToast("Allow pop-ups to open Print / PDF view");

    printWin.document.write(`<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>${escapeHtml(songName)} - Lead Sheet</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Poppins', 'Nirmala UI', 'Segoe UI', Arial, sans-serif;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  body { background: #f1f5f9; color: #0f172a; padding: 28px; }
  .sheet {
    max-width: 820px; margin: 0 auto; background: #ffffff; padding: 38px 44px;
    border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  }
  .top-actions {
    max-width: 820px; margin: 0 auto 14px; display: flex; justify-content: flex-end;
  }
  .print-btn {
    background: #f97316; color: #fff; border: none; padding: 10px 18px; border-radius: 8px;
    font-weight: 600; font-size: 14px; cursor: pointer;
  }
  .header {
    border-bottom: 2px solid #0f172a; padding-bottom: 14px; margin-bottom: 26px;
    display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap;
  }
  .song-name { font-size: 21px; font-weight: 600; color: #0f172a; max-width: 520px; letter-spacing: -0.2px; }
  .meta-group { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    font-size: 12px; font-weight: 600; padding: 4px 11px; border-radius: 6px;
    background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b;
  }
  .pill.capo { background: #fff7ed; border-color: #fdba74; color: #c2410c; }
  .lyric-line { display: flex; flex-wrap: wrap; margin-bottom: 14px; page-break-inside: avoid; }
  .tok { display: inline-flex; flex-direction: column; margin-right: 3px; }
  .ch { font-size: 14px; font-weight: 700; color: #c2410c; min-height: 20px; letter-spacing: 0.2px; }
  .wd { font-size: 16px; font-weight: 500; color: #0f172a; line-height: 1.5; }
  .spacer { height: 18px; }
  @media print {
    body { background: #ffffff; padding: 0; }
    .top-actions { display: none; }
    .sheet { box-shadow: none; padding: 10px; max-width: 100%; }
  }
</style>
</head>
<body>
  <div class="top-actions">
    <button class="print-btn" onclick="window.print()">Print / Save Lead Sheet PDF</button>
  </div>
  <div class="sheet">
    <div class="header">
      <div class="song-name">${escapeHtml(songName)}</div>
      <div class="meta-group">
        <span class="pill">Key: ${escapeHtml(tKey)}</span>
        ${songBpm > 0 ? `<span class="pill">${songBpm} BPM</span>` : ""}
        <span class="pill">${escapeHtml(songTimeSig)}</span>
        ${capoBadge}
      </div>
    </div>
    ${renderedLines}
  </div>
</body>
</html>`);
    printWin.document.close();
  }

  document.getElementById("ls-print-btn").onclick = printLeadSheet;
  document.getElementById("ls-bar-print-btn").onclick = printLeadSheet;

  // ============================================================================
  // CIRCLE OF FIFTHS, DIAGRAMS & PLAYBACK SYNC
  // ============================================================================
  function getKeyFamily(keyLabel) {
    const [root, qual] = (keyLabel || "C Major").split(" ");
    let majorRoot = root;
    if (qual === "Minor") {
      const rIdx = PITCHES.indexOf(root);
      majorRoot = rIdx !== -1 ? PITCHES[(rIdx + 3) % 12] : "C";
    }
    const cIdx = CIRCLE_MAJOR.indexOf(majorRoot);
    const center = cIdx !== -1 ? cIdx : 0;
    const leftIdx = (center + 11) % 12;
    const rightIdx = (center + 1) % 12;

    const familySet = new Set([
      CIRCLE_MAJOR[leftIdx], CIRCLE_MAJOR[center], CIRCLE_MAJOR[rightIdx],
      CIRCLE_MINOR[leftIdx], CIRCLE_MINOR[center], CIRCLE_MINOR[rightIdx]
    ]);

    const orderedList = qual === "Minor"
      ? [
          { short: CIRCLE_MINOR[center], role: "i", tonic: true },
          { short: CIRCLE_MAJOR[center], role: "III", tonic: false },
          { short: CIRCLE_MINOR[leftIdx], role: "iv", tonic: false },
          { short: CIRCLE_MAJOR[leftIdx], role: "VI", tonic: false },
          { short: CIRCLE_MINOR[rightIdx], role: "v", tonic: false },
          { short: CIRCLE_MAJOR[rightIdx], role: "VII", tonic: false }
        ]
      : [
          { short: CIRCLE_MAJOR[center], role: "I", tonic: true },
          { short: CIRCLE_MINOR[leftIdx], role: "ii", tonic: false },
          { short: CIRCLE_MINOR[rightIdx], role: "iii", tonic: false },
          { short: CIRCLE_MAJOR[leftIdx], role: "IV", tonic: false },
          { short: CIRCLE_MAJOR[rightIdx], role: "V", tonic: false },
          { short: CIRCLE_MINOR[center], role: "vi", tonic: false }
        ];

    const tonicShort = qual === "Minor" ? `${root}m` : root;
    return { familySet, orderedList, tonicShort };
  }

  function renderCircleOfFifths(keyLabel) {
    const tagEl = document.getElementById("circle-key-tag");
    const circleContainer = document.getElementById("circle-container");
    const chipsContainer = document.getElementById("scale-chips");

    tagEl.textContent = keyLabel;
    const { familySet, orderedList, tonicShort } = getKeyFamily(keyLabel);

    const size = 196;
    const cx = size / 2, cy = size / 2;
    const rOuter = 82, rInner = 52;

    let svg = `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">`;
    svg += `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="none" stroke="#222a35" stroke-width="24"/>`;
    svg += `<circle cx="${cx}" cy="${cy}" r="${rInner}" fill="none" stroke="#192029" stroke-width="20"/>`;
    svg += `<circle cx="${cx}" cy="${cy}" r="35" fill="#141920" stroke="#262f3b" stroke-width="1.2"/>`;
    svg += `<text x="${cx}" y="${cy - 3}" text-anchor="middle" fill="#f8fafc" font-size="11.5" font-weight="800">${escapeHtml(toShortChord(keyLabel))}</text>`;
    svg += `<text x="${cx}" y="${cy + 10}" text-anchor="middle" fill="#64748b" font-size="8.5" font-weight="700">KEY FAMILY</text>`;

    for (let i = 0; i < 12; i++) {
      const angle = (i * 30 - 90) * (Math.PI / 180);
      const majName = CIRCLE_MAJOR[i];
      const minName = CIRCLE_MINOR[i];

      const majInKey = familySet.has(majName);
      const minInKey = familySet.has(minName);
      const majIsTonic = majName === tonicShort;
      const minIsTonic = minName === tonicShort;

      const mx = cx + rOuter * Math.cos(angle);
      const my = cy + rOuter * Math.sin(angle);

      const ix = cx + rInner * Math.cos(angle);
      const iy = cy + rInner * Math.sin(angle);

      const majFill = majIsTonic ? "#f97316" : (majInKey ? "#1e3a5f" : "transparent");
      const majStroke = majIsTonic ? "#fdba74" : (majInKey ? "#38bdf8" : "none");
      const majText = majIsTonic ? "#111827" : (majInKey ? "#f8fafc" : "#64748b");
      svg += `<g style="cursor:pointer" onclick="previewChordFromWheel('${majName}')">
        <circle cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="12" fill="${majFill}" stroke="${majStroke}" stroke-width="1.3"/>
        <text x="${mx.toFixed(1)}" y="${(my + 3.8).toFixed(1)}" text-anchor="middle" fill="${majText}" font-size="9.5" font-weight="800">${majName}</text>
      </g>`;

      const minFill = minIsTonic ? "#f97316" : (minInKey ? "#1e3a5f" : "transparent");
      const minStroke = minIsTonic ? "#fdba74" : (minInKey ? "#38bdf8" : "none");
      const minText = minIsTonic ? "#111827" : (minInKey ? "#e2e8f0" : "#525e6e");
      svg += `<g style="cursor:pointer" onclick="previewChordFromWheel('${minName}')">
        <circle cx="${ix.toFixed(1)}" cy="${iy.toFixed(1)}" r="10.2" fill="${minFill}" stroke="${minStroke}" stroke-width="1.2"/>
        <text x="${ix.toFixed(1)}" y="${(iy + 3.2).toFixed(1)}" text-anchor="middle" fill="${minText}" font-size="8.2" font-weight="700">${minName}</text>
      </g>`;
    }
    svg += `</svg>`;
    circleContainer.innerHTML = svg;

    chipsContainer.innerHTML = orderedList.map(item => `
      <span class="scale-chip ${item.tonic ? 'tonic' : ''}"
            title="Click to preview ${item.short} diagram (${item.role})"
            onclick="previewChordFromWheel('${item.short}')">
        ${item.short}
      </span>
    `).join("");
  }

  window.previewChordFromWheel = function(shortChord) {
    const full = shortToFullChord(shortChord);
    currentFullChord = full;
    renderChordDiagram(full);
  };

  function renderGuitarSVG(chordLabel) {
    const frets = GUITAR_VOICINGS[chordLabel];
    const w = 136, h = 122;
    const padL = 22, padT = 20, cols = 5, rows = 4;
    const colW = 19, rowH = 22;

    if (!frets) {
      return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
        <text x="50%" y="54%" text-anchor="middle" fill="#64748b" font-size="12">No Diagram</text>
      </svg>`;
    }

    const played = frets.filter(f => f > 0);
    const maxFret = played.length ? Math.max(...played) : 0;
    const baseFret = maxFret > 4 ? Math.min(...played) : 1;

    let svg = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`;
    if (baseFret === 1) {
      svg += `<line x1="${padL}" y1="${padT}" x2="${padL + cols * colW}" y2="${padT}" stroke="#cbd5e1" stroke-width="3.5"/>`;
    } else {
      svg += `<text x="${padL - 7}" y="${padT + 15}" text-anchor="end" fill="#fb923c" font-size="11" font-weight="700">${baseFret}fr</text>`;
    }

    for (let r = 0; r <= rows; r++) {
      const y = padT + r * rowH;
      svg += `<line x1="${padL}" y1="${y}" x2="${padL + cols * colW}" y2="${y}" stroke="#334155" stroke-width="1.2"/>`;
    }
    for (let c = 0; c <= cols; c++) {
      const x = padL + c * colW;
      svg += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + rows * rowH}" stroke="#475569" stroke-width="1.2"/>`;
    }

    frets.forEach((f, idx) => {
      const x = padL + idx * colW;
      if (f === -1) {
        svg += `<text x="${x}" y="${padT - 6}" text-anchor="middle" fill="#64748b" font-size="11" font-weight="700">&times;</text>`;
      } else if (f === 0) {
        svg += `<circle cx="${x}" cy="${padT - 9}" r="3.5" fill="none" stroke="#94a3b8" stroke-width="1.4"/>`;
      } else {
        const relFret = f - baseFret + 1;
        if (relFret >= 1 && relFret <= rows) {
          const cy = padT + (relFret - 0.5) * rowH;
          svg += `<circle cx="${x}" cy="${cy}" r="6.5" fill="#f97316"/>`;
        }
      }
    });
    svg += `</svg>`;
    return svg;
  }

  function renderPianoSVG(chordLabel) {
    const w = 154, h = 92;
    let activeNotes = new Set();
    if (chordLabel && chordLabel !== "N") {
      const [root, qual] = chordLabel.split(" ");
      const rIdx = PITCHES.indexOf(root);
      if (rIdx !== -1) {
        const intervals = qual === "Minor" ? [0, 3, 7] : [0, 4, 7];
        intervals.forEach(iv => activeNotes.add((rIdx + iv) % 12));
      }
    }

    const whiteSemis = [0, 2, 4, 5, 7, 9, 11];
    const blackSemis = [
      { semi: 1, afterWhite: 0 }, { semi: 3, afterWhite: 1 },
      { semi: 6, afterWhite: 3 }, { semi: 8, afterWhite: 4 }, { semi: 10, afterWhite: 5 }
    ];
    const kw = w / 7;
    let svg = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`;
    whiteSemis.forEach((semi, i) => {
      const active = activeNotes.has(semi);
      svg += `<rect x="${i * kw + 1}" y="2" width="${kw - 2}" height="${h - 4}" rx="3"
        fill="${active ? '#f97316' : '#e2e8f0'}" stroke="#0f1318" stroke-width="1.2"/>`;
    });
    blackSemis.forEach(b => {
      const active = activeNotes.has(b.semi);
      const bx = (b.afterWhite + 1) * kw - (kw * 0.32);
      svg += `<rect x="${bx}" y="2" width="${kw * 0.64}" height="${h * 0.58}" rx="2.5"
        fill="${active ? '#fb923c' : '#1e293b'}" stroke="#0f1318" stroke-width="1.2"/>`;
    });
    svg += `</svg>`;
    return svg;
  }

  function renderChordDiagram(chordLabel) {
    const container = document.getElementById("diagram-container");
    container.innerHTML = diagramMode === "guitar"
      ? renderGuitarSVG(chordLabel)
      : renderPianoSVG(chordLabel);
  }

  function ordinal(n) {
    if (n === 1) return "1st";
    if (n === 2) return "2nd";
    if (n === 3) return "3rd";
    return `${n}th`;
  }

  function updateCapoBadge() {
    const badge = document.getElementById("capo-badge");
    if (transposeSteps < 0) {
      badge.style.display = "inline-block";
      badge.textContent = `Capo ${ordinal(Math.abs(transposeSteps))} Fret`;
    } else if (transposeSteps > 0) {
      badge.style.display = "inline-block";
      badge.textContent = `+${transposeSteps} Pitch Shift`;
    } else {
      badge.style.display = "none";
    }
  }

  function updateMetaHeader() {
    const tKey = transposeLabel(songKey, transposeSteps);
    renderCircleOfFifths(tKey);
    if (songBpm > 0) {
      const groupEl = document.getElementById("meta-group-top");
      const metaEl = document.getElementById("song-meta");
      const sigEl = document.getElementById("sig-pill");
      groupEl.style.display = "inline-flex";
      metaEl.textContent = `${tKey} • ${songBpm} BPM`;
      sigEl.textContent = songTimeSig;
    }
    if (currentRoute === "leadsheet") {
      document.getElementById("ls-song-badge").textContent =
        rawChords.length ? `${songName} • ${tKey} • ${songTimeSig}` : "Ready (Paste Lyrics)";
      renderLeadSheetSequenceStrip();
      renderLeadSheetPalette();
    }
  }

  // Open Modern Center Modal for Time Signature Change
  const sigModal = document.getElementById("sig-modal");
  document.getElementById("sig-pill").onclick = () => {
    const idx = TIME_SIGS.indexOf(songTimeSig);
    pendingTimeSig = TIME_SIGS[(idx + 1) % TIME_SIGS.length];
    document.getElementById("sig-old-val").textContent = songTimeSig;
    document.getElementById("sig-new-val").textContent = pendingTimeSig;
    sigModal.classList.add("open");
  };

  document.getElementById("btn-sig-cancel").onclick = () => {
    sigModal.classList.remove("open");
  };
  sigModal.onclick = (e) => {
    if (e.target === sigModal) sigModal.classList.remove("open");
  };

  document.getElementById("btn-sig-confirm").onclick = async () => {
    sigModal.classList.remove("open");
    const nextSig = pendingTimeSig;
    songTimeSig = nextSig;
    updateMetaHeader();

    if (!currentAudioId || !rawChords.length) return;

    const useDemucs = document.getElementById("demucs-toggle").checked;
    const loader = document.getElementById("loader");
    const startTime = performance.now();
    loader.style.display = "flex";
    updateProgressModal({
      percent: 45,
      stage: `Re-aligning Chords to ${nextSig} Meter...`,
      detail: `Locking downbeats & bar transitions for ${nextSig} feel...`,
      eta: "0:02",
      speed: `${nextSig} Grid`,
      speedLbl: "Meter Lock",
      step: 3,
      useDemucs
    });

    const elapsedTimer = setInterval(() => {
      const sec = Math.floor((performance.now() - startTime) / 1000);
      document.getElementById("stat-elapsed").textContent = fmtTime(sec);
    }, 500);

    try {
      const formData = new FormData();
      formData.append("audio_id", currentAudioId);
      formData.append("time_signature", nextSig);
      formData.append("use_demucs", useDemucs);

      const res = await fetch(`/api/realign_meter?t=${Date.now()}`, {
        method: "POST",
        body: formData,
        cache: "no-store"
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Meter re-alignment failed");

      rawChords = data.chords;
      songKey = data.key;
      songBpm = data.bpm;
      songTimeSig = data.time_signature || nextSig;
      renderTimeline();
      showToast(`Chords re-aligned to ${songTimeSig} meter`);
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      clearInterval(elapsedTimer);
      loader.style.display = "none";
    }
  };

  document.getElementById("bpm-half").onclick = () => {
    if (songBpm > 30) {
      songBpm = Math.round((songBpm / 2) * 10) / 10;
      updateMetaHeader();
      showToast(`Tempo adjusted to ${songBpm} BPM`);
    }
  };
  document.getElementById("bpm-double").onclick = () => {
    if (songBpm > 0 && songBpm < 150) {
      songBpm = Math.round((songBpm * 2) * 10) / 10;
      updateMetaHeader();
      showToast(`Tempo adjusted to ${songBpm} BPM`);
    }
  };

  // Modern Chord Edit / Global Replace Modal
  const editModal = document.getElementById("edit-modal");
  function openChordEditor(idx) {
    if (idx < 0 || idx >= rawChords.length) return;
    editingBlockIdx = idx;
    const origTransposed = transposeLabel(rawChords[idx].chord, transposeSteps);
    document.getElementById("edit-title").textContent = `Edit "${toShortChord(origTransposed)}" (${fmtTime(rawChords[idx].start)})`;
    document.getElementById("edit-select").value = origTransposed;
    editModal.classList.add("open");
  }

  document.getElementById("btn-edit-cancel").onclick = () => {
    editModal.classList.remove("open");
  };
  editModal.onclick = (e) => {
    if (e.target === editModal) editModal.classList.remove("open");
  };

  document.getElementById("btn-replace-one").onclick = () => {
    if (editingBlockIdx < 0) return;
    const chosenTransposed = document.getElementById("edit-select").value;
    rawChords[editingBlockIdx].chord = transposeLabel(chosenTransposed, -transposeSteps);
    editModal.classList.remove("open");
    renderTimeline();
    showToast(`Updated block to ${toShortChord(chosenTransposed)}`);
  };

  document.getElementById("btn-replace-all").onclick = () => {
    if (editingBlockIdx < 0) return;
    const targetRawChord = rawChords[editingBlockIdx].chord;
    const chosenTransposed = document.getElementById("edit-select").value;
    const newRawChord = transposeLabel(chosenTransposed, -transposeSteps);

    let count = 0;
    rawChords.forEach(item => {
      if (item.chord === targetRawChord) {
        item.chord = newRawChord;
        count++;
      }
    });
    editModal.classList.remove("open");
    renderTimeline();
    showToast(`Replaced ${count} occurrences with ${toShortChord(chosenTransposed)}`);
  };

  function renderTimeline() {
    const track = document.getElementById("timeline-track");
    track.innerHTML = "";
    rawChords.forEach((item, idx) => {
      const tLabel = transposeLabel(item.chord, transposeSteps);
      const shortName = toShortChord(tLabel);
      const dur = Math.max(0.32, item.end - item.start);
      const widthPx = Math.min(260, Math.max(36, Math.round(dur * 28)));

      const block = document.createElement("div");
      block.className = "chord-block";
      block.dataset.idx = idx;
      block.style.width = `${widthPx}px`;
      block.style.background = getRootColor(tLabel);
      block.textContent = shortName;
      block.title = `${fmtTime(item.start)} - ${fmtTime(item.end)} | ${tLabel} (Double-click to edit)`;
      block.onclick = () => {
        audio.currentTime = item.start + 0.005;
        if (audio.paused) audio.play();
        syncPlaybackUI(true);
      };
      block.ondblclick = (e) => {
        e.stopPropagation();
        openChordEditor(idx);
      };
      track.appendChild(block);
    });

    updateMetaHeader();
    activeIndex = -1;
    syncPlaybackUI(true);
  }

  function syncPlaybackUI(force = false) {
    const rawTime = audio.currentTime || 0;
    const t = rawTime + 0.06;

    timeDisplay.textContent = `${fmtTime(rawTime)} / ${fmtTime(audio.duration || 0)}`;

    if (!seekBar.matches(":active") && isFinite(audio.duration)) {
      seekBar.value = rawTime;
    }

    if (!rawChords.length) {
      renderChordDiagram("N");
      return;
    }

    let found = -1;
    for (let i = 0; i < rawChords.length; i++) {
      if (t >= rawChords[i].start && t <= rawChords[i].end) {
        found = i;
        break;
      }
    }
    if (found === -1) {
      found = t < rawChords[0].start ? 0 : rawChords.length - 1;
    }

    if (found !== -1 && (found !== activeIndex || force)) {
      activeIndex = found;
      const currLabel = transposeLabel(rawChords[found].chord, transposeSteps);
      const nextLabel = (found + 1 < rawChords.length)
        ? transposeLabel(rawChords[found + 1].chord, transposeSteps)
        : "—";

      const shortCurr = toShortChord(currLabel);
      const shortNext = toShortChord(nextLabel);

      currentFullChord = currLabel;
      const shortEl = document.getElementById("chord-short");
      shortEl.textContent = shortCurr;
      shortEl.style.color = currLabel === "N" ? "#64748b" : "#a39e3b";
      document.getElementById("chord-full").textContent = currLabel === "N" ? "No Chord" : currLabel;
      document.getElementById("chord-next").textContent = shortNext;
      renderChordDiagram(currLabel);

      // Update Lead Sheet Current & Next Chord Pills above Lyric Canvas
      document.getElementById("ls-curr-chord").textContent = shortCurr;
      document.getElementById("ls-next-chord").textContent = shortNext;

      const blocks = document.querySelectorAll(".chord-block");
      blocks.forEach(b => b.classList.remove("active"));
      if (blocks[found]) {
        blocks[found].classList.add("active");
        if (currentRoute === "detector") {
          blocks[found].scrollIntoView({ behavior: "auto", inline: "center", block: "nearest" });
        }
      }

      if (currentRoute === "leadsheet") {
        const seqItems = lsSeqStrip.querySelectorAll(".ls-seq-item");
        seqItems.forEach(el => el.classList.remove("active"));
        if (seqItems[found]) {
          seqItems[found].classList.add("active");
          seqItems[found].scrollIntoView({ behavior: "auto", inline: "center", block: "nearest" });
        }
      }
    }
  }

  function startRenderLoop() {
    if (rafId) cancelAnimationFrame(rafId);
    const tick = () => {
      syncPlaybackUI(false);
      if (!audio.paused && !audio.ended) {
        rafId = requestAnimationFrame(tick);
      }
    };
    rafId = requestAnimationFrame(tick);
  }

  document.getElementById("tab-guitar").onclick = () => {
    diagramMode = "guitar";
    document.getElementById("tab-guitar").classList.add("active");
    document.getElementById("tab-piano").classList.remove("active");
    renderChordDiagram(currentFullChord);
  };
  document.getElementById("tab-piano").onclick = () => {
    diagramMode = "piano";
    document.getElementById("tab-piano").classList.add("active");
    document.getElementById("tab-guitar").classList.remove("active");
    renderChordDiagram(currentFullChord);
  };

  async function handleFile(file) {
    if (!file) return;
    songName = file.name.replace(/\.[^/.]+$/, "");
    document.getElementById("song-title").textContent = songName;
    resetAudioSource(URL.createObjectURL(file));

    const useDemucs = document.getElementById("demucs-toggle").checked;
    const jobId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const loader = document.getElementById("loader");
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
    const startTime = performance.now();

    loader.style.display = "flex";
    updateProgressModal({
      percent: 2,
      stage: "Uploading Audio File...",
      detail: `${file.name} (${fileSizeMB} MB)`,
      eta: "0:01",
      speed: "Starting...",
      speedLbl: "Upload Speed",
      step: 1,
      useDemucs
    });

    const elapsedTimer = setInterval(() => {
      const sec = Math.floor((performance.now() - startTime) / 1000);
      document.getElementById("stat-elapsed").textContent = fmtTime(sec);
    }, 500);

    let pollTimer = null;
    const startPolling = () => {
      pollTimer = setInterval(async () => {
        try {
          const pRes = await fetch(`/api/progress/${jobId}?t=${Date.now()}`, { cache: "no-store" });
          if (pRes.ok) {
            const p = await pRes.json();
            updateProgressModal({
              percent: p.percent,
              stage: p.stage,
              detail: p.detail,
              eta: p.eta,
              speed: useDemucs ? "AI GPU/CPU" : "High-Res CQT",
              speedLbl: "Engine Mode",
              step: p.step,
              useDemucs
            });
          }
        } catch (_) {}
      }, 650);
    };

    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append("file", file);
        formData.append("use_demucs", useDemucs);
        formData.append("job_id", jobId);

        xhr.open("POST", `/api/analyze?t=${Date.now()}`, true);

        xhr.upload.onprogress = (evt) => {
          if (evt.lengthComputable) {
            const uploadFrac = evt.loaded / evt.total;
            const pct = Math.min(16, Math.max(2, Math.round(uploadFrac * 16)));
            const elapsedSec = Math.max(0.05, (performance.now() - startTime) / 1000);
            const mbPerSec = ((evt.loaded / (1024 * 1024)) / elapsedSec).toFixed(1);
            updateProgressModal({
              percent: pct,
              stage: `Uploading Audio (${Math.round(uploadFrac * 100)}%)...`,
              detail: `Sent ${(evt.loaded / (1024 * 1024)).toFixed(2)} MB of ${fileSizeMB} MB`,
              speed: `${mbPerSec} MB/s`,
              speedLbl: "Upload Speed",
              step: 1,
              useDemucs
            });
            if (evt.loaded >= evt.total && !pollTimer) startPolling();
          }
        };

        xhr.upload.onload = () => { if (!pollTimer) startPolling(); };

        xhr.onload = () => {
          try {
            const parsed = JSON.parse(xhr.responseText);
            if (xhr.status >= 200 && xhr.status < 300) resolve(parsed);
            else reject(new Error(parsed.detail || "Analysis failed"));
          } catch (e) {
            reject(new Error("Invalid server response"));
          }
        };

        xhr.onerror = () => reject(new Error("Network error while communicating with local server"));
        xhr.send(formData);
      });

      rawChords = data.chords;
      songKey = data.key;
      songBpm = data.bpm;
      songTimeSig = data.time_signature || "4/4";
      currentAudioId = data.audio_id || "";
      renderTimeline();
      showToast(`Detected ${rawChords.length} chords in ${songKey}`);
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      clearInterval(elapsedTimer);
      if (pollTimer) clearInterval(pollTimer);
      loader.style.display = "none";
    }
  }

  async function handleUrlAnalyze() {
    const urlInput = document.getElementById("url-input");
    const url = urlInput.value.trim();
    if (!url) return showToast("Paste a YouTube or audio link first");

    const useDemucs = document.getElementById("demucs-toggle").checked;
    const jobId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const loader = document.getElementById("loader");
    const startTime = performance.now();

    loader.style.display = "flex";
    updateProgressModal({
      percent: 3,
      stage: "Connecting to Link...",
      detail: url,
      eta: "0:15",
      speed: "Fetching...",
      speedLbl: "Download Speed",
      step: 1,
      useDemucs
    });

    const elapsedTimer = setInterval(() => {
      const sec = Math.floor((performance.now() - startTime) / 1000);
      document.getElementById("stat-elapsed").textContent = fmtTime(sec);
    }, 500);

    const pollTimer = setInterval(async () => {
      try {
        const pRes = await fetch(`/api/progress/${jobId}?t=${Date.now()}`, { cache: "no-store" });
        if (pRes.ok) {
          const p = await pRes.json();
          updateProgressModal({
            percent: p.percent,
            stage: p.stage,
            detail: p.detail,
            eta: p.eta,
            speed: p.speed || (useDemucs ? "AI GPU/CPU" : "High-Res CQT"),
            speedLbl: p.step === 1 ? "Download Speed" : "Engine Mode",
            step: p.step,
            useDemucs
          });
        }
      } catch (_) {}
    }, 600);

    try {
      const formData = new FormData();
      formData.append("url", url);
      formData.append("use_demucs", useDemucs);
      formData.append("job_id", jobId);

      const res = await fetch(`/api/analyze_url?t=${Date.now()}`, {
        method: "POST",
        body: formData,
        cache: "no-store"
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Link analysis failed");

      songName = data.song_name || "YouTube Audio";
      document.getElementById("song-title").textContent = songName;
      if (data.audio_url) {
        resetAudioSource(data.audio_url);
      }

      rawChords = data.chords;
      songKey = data.key;
      songBpm = data.bpm;
      songTimeSig = data.time_signature || "4/4";
      currentAudioId = data.audio_id || "";
      renderTimeline();
      showToast(`Detected ${rawChords.length} chords in ${songKey}`);
    } catch (err) {
      showToast("Error: " + err.message);
    } finally {
      clearInterval(elapsedTimer);
      clearInterval(pollTimer);
      loader.style.display = "none";
    }
  }

  document.getElementById("url-btn").addEventListener("click", handleUrlAnalyze);
  document.getElementById("url-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleUrlAnalyze();
    }
  });

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function togglePlayPause() {
    if (!audio.src) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }
  function skipSeconds(delta) {
    if (!audio.src) return;
    const maxDur = isFinite(audio.duration) ? audio.duration : 999999;
    audio.currentTime = Math.max(0, Math.min(maxDur, (audio.currentTime || 0) + delta));
    syncPlaybackUI(true);
  }

  playBtn.onclick = togglePlayPause;
  skipBackBtn.onclick = () => skipSeconds(-5);
  skipFwdBtn.onclick = () => skipSeconds(5);

  audio.onplay = () => {
    playBtn.innerHTML = "&#10074;&#10074;";
    startRenderLoop();
  };
  audio.onpause = () => {
    playBtn.innerHTML = "&#9654;";
    if (rafId) cancelAnimationFrame(rafId);
    syncPlaybackUI(true);
  };
  audio.onended = () => {
    playBtn.innerHTML = "&#9654;";
    if (rafId) cancelAnimationFrame(rafId);
  };
  audio.onloadedmetadata = () => {
    seekBar.max = audio.duration;
    timeDisplay.textContent = `0:00 / ${fmtTime(audio.duration)}`;
  };
  seekBar.oninput = () => { audio.currentTime = seekBar.value; syncPlaybackUI(true); };

  function updateTranspose(delta) {
    transposeSteps = Math.max(-12, Math.min(12, transposeSteps + delta));
    document.getElementById("trans-val").textContent = (transposeSteps >= 0 ? "+" : "") + transposeSteps;
    updateCapoBadge();
    renderTimeline();
  }
  document.getElementById("trans-down").onclick = () => updateTranspose(-1);
  document.getElementById("trans-up").onclick = () => updateTranspose(1);

  document.addEventListener("keydown", (e) => {
    if (e.code === "Escape") {
      sigModal.classList.remove("open");
      editModal.classList.remove("open");
      return;
    }
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) && document.activeElement.type !== "range") {
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      skipSeconds(-5);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      skipSeconds(5);
    } else if (e.code === "ArrowUp") {
      e.preventDefault();
      updateTranspose(1);
    } else if (e.code === "ArrowDown") {
      e.preventDefault();
      updateTranspose(-1);
    }
  });

  function downloadFile(content, filename, mime) {
    if (!rawChords.length) return showToast("Upload and analyze a song first");
    const blob = new Blob([content], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  }

  document.getElementById("dl-txt").onclick = () => {
    const tKey = transposeLabel(songKey, transposeSteps);
    const capoNote = transposeSteps < 0 ? ` | Capo ${ordinal(Math.abs(transposeSteps))} Fret` : "";
    const header = `${songName} | Key: ${tKey} | ${songBpm} BPM | Time: ${songTimeSig}${capoNote}\n` + "-".repeat(54) + "\n";
    const lines = rawChords.map(c => `${fmtTime(c.start)} - ${fmtTime(c.end)} | ${transposeLabel(c.chord, transposeSteps)}`);
    downloadFile(header + lines.join("\n"), `${songName}_chords.txt`, "text/plain");
  };

  document.getElementById("dl-csv").onclick = () => {
    const lines = ["Start Time,End Time,Chord"];
    rawChords.forEach(c => {
      lines.push(`${fmtTime(c.start)},${fmtTime(c.end)},"${transposeLabel(c.chord, transposeSteps)}"`);
    });
    downloadFile(lines.join("\n"), `${songName}_chords.csv`, "text/csv");
  };

  document.getElementById("dl-pdf").onclick = () => {
    if (!rawChords.length) return showToast("Upload and analyze a song first");

    const tKey = transposeLabel(songKey, transposeSteps);
    let capoBadge = "";
    if (transposeSteps < 0) {
      capoBadge = `<span class="pill capo">Capo ${ordinal(Math.abs(transposeSteps))} Fret</span>`;
    } else if (transposeSteps > 0) {
      capoBadge = `<span class="pill capo">Transpose +${transposeSteps}</span>`;
    }

    const barsHtml = rawChords.map((c) => {
      const full = transposeLabel(c.chord, transposeSteps);
      const shortC = toShortChord(full);
      return `
        <div class="bar-cell">
          <div class="bar-time">${fmtTime(c.start)} &ndash; ${fmtTime(c.end)}</div>
          <div class="bar-chord">${escapeHtml(shortC)}</div>
          <div class="bar-sub">${escapeHtml(full)}</div>
        </div>
      `;
    }).join("");

    const printWin = window.open("", "_blank", "width=920,height=760");
    if (!printWin) return showToast("Allow pop-ups to open Print / PDF view");

    printWin.document.write(`<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>${escapeHtml(songName)} - Chord Chart</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Poppins', 'Segoe UI', Arial, sans-serif; }
  body { background: #f1f5f9; color: #0f172a; padding: 28px; }
  .sheet {
    max-width: 820px; margin: 0 auto; background: #ffffff; padding: 34px 38px;
    border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
  }
  .top-actions {
    max-width: 820px; margin: 0 auto 14px; display: flex; justify-content: flex-end; gap: 10px;
  }
  .print-btn {
    background: #f97316; color: #fff; border: none; padding: 10px 18px; border-radius: 8px;
    font-weight: 600; font-size: 14px; cursor: pointer;
  }
  .print-btn:hover { background: #ea580c; }
  .header {
    border-bottom: 2px solid #0f172a; padding-bottom: 14px; margin-bottom: 22px;
    display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap;
  }
  .song-name { font-size: 21px; font-weight: 600; color: #0f172a; max-width: 520px; }
  .meta-group { display: flex; gap: 8px; flex-wrap: wrap; }
  .pill {
    font-size: 12px; font-weight: 600; padding: 5px 11px; border-radius: 6px;
    background: #f1f5f9; border: 1px solid #cbd5e1; color: #1e293b;
  }
  .pill.capo { background: #fff7ed; border-color: #fdba74; color: #c2410c; }
  .grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .bar-cell {
    border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 10px 12px;
    background: #f8fafc; page-break-inside: avoid;
  }
  .bar-time { font-size: 11px; color: #64748b; font-weight: 500; margin-bottom: 4px; }
  .bar-chord { font-size: 23px; font-weight: 700; color: #0f172a; line-height: 1.15; }
  .bar-sub { font-size: 11.5px; font-weight: 500; color: #475569; margin-top: 2px; }
  .footer {
    margin-top: 24px; padding-top: 10px; border-top: 1px solid #e2e8f0;
    font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between;
  }
  @media print {
    body { background: #ffffff; padding: 0; }
    .top-actions { display: none; }
    .sheet { box-shadow: none; padding: 10px; max-width: 100%; }
  }
</style>
</head>
<body>
  <div class="top-actions">
    <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <div class="sheet">
    <div class="header">
      <div class="song-name">${escapeHtml(songName)}</div>
      <div class="meta-group">
        <span class="pill">Key: ${escapeHtml(tKey)}</span>
        <span class="pill">${songBpm} BPM</span>
        <span class="pill">${escapeHtml(songTimeSig)}</span>
        ${capoBadge}
      </div>
    </div>
    <div class="grid">${barsHtml}</div>
    <div class="footer">
      <span>Generated by Offline Chord Detector</span>
      <span>Total Bars / Segments: ${rawChords.length}</span>
    </div>
  </div>
</body>
</html>`);
    printWin.document.close();
  };

  renderChordDiagram("N");
  renderCircleOfFifths("C Major");
</script>
</body>
</html>
"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_UI, headers=NO_CACHE_HEADERS)


@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    data = JOBS_PROGRESS.get(job_id, {
        "percent": 10,
        "stage": "Initializing...",
        "detail": "Preparing audio analysis...",
        "eta": "--",
        "speed": "--",
        "step": 1,
        "done": False,
    })
    return JSONResponse(data, headers=NO_CACHE_HEADERS)


@app.get("/api/cached_audio/{filename}")
def serve_cached_audio(filename: str):
    safe_name = Path(filename).name
    target = CACHE_AUDIO_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(target), headers=NO_CACHE_HEADERS)


@app.post("/api/analyze")
def api_analyze(
    file: UploadFile = File(...),
    use_demucs: bool = Form(False),
    job_id: str = Form(""),
):
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        update_job(
            job_id,
            percent=16,
            step=1,
            stage="Preparing Fresh Audio Analysis...",
            detail="Staging audio file...",
            eta="0:03",
        )
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_id = compute_short_id(temp_path.read_bytes())
        staged_audio = UPLOAD_DIR / f"active_{audio_id}{suffix}"
        shutil.copy2(temp_path, staged_audio)

        result = analyze_audio_file(str(staged_audio), use_demucs=use_demucs, job_id=job_id, forced_time_sig="auto")
        result["song_name"] = file.filename or "Unknown"
        result["audio_id"] = f"upload:{staged_audio.name}"
        return JSONResponse(result, headers=NO_CACHE_HEADERS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if job_id in JOBS_PROGRESS:
            JOBS_PROGRESS.pop(job_id, None)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/api/analyze_url")
def api_analyze_url(
    url: str = Form(...),
    use_demucs: bool = Form(False),
    job_id: str = Form(""),
):
    clean_url = url.strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        url_id = compute_short_id(clean_url.encode("utf-8"))
        with tempfile.TemporaryDirectory() as tmp_dl:
            dl_path, title = download_audio_from_url(clean_url, Path(tmp_dl), job_id=job_id)
            saved_audio_name = f"url_{url_id}{dl_path.suffix or '.m4a'}"
            saved_audio_path = CACHE_AUDIO_DIR / saved_audio_name
            shutil.copy2(dl_path, saved_audio_path)

            result = analyze_audio_file(str(saved_audio_path), use_demucs=use_demucs, job_id=job_id, forced_time_sig="auto")
            result["song_name"] = title
            result["audio_url"] = f"/api/cached_audio/{saved_audio_name}"
            result["audio_id"] = f"cached:{saved_audio_name}"
            return JSONResponse(result, headers=NO_CACHE_HEADERS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if job_id in JOBS_PROGRESS:
            JOBS_PROGRESS.pop(job_id, None)


@app.post("/api/realign_meter")
def api_realign_meter(
    audio_id: str = Form(...),
    time_signature: str = Form(...),
    use_demucs: bool = Form(False),
):
    if time_signature not in ("4/4", "3/4", "6/8"):
        raise HTTPException(status_code=400, detail="Unsupported time signature")

    if audio_id.startswith("cached:"):
        fname = Path(audio_id.split(":", 1)[1]).name
        target_path = CACHE_AUDIO_DIR / fname
    elif audio_id.startswith("upload:"):
        fname = Path(audio_id.split(":", 1)[1]).name
        target_path = UPLOAD_DIR / fname
    else:
        raise HTTPException(status_code=400, detail="Invalid audio reference")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Audio file expired; please re-upload or re-paste link.")

    try:
        result = analyze_audio_file(str(target_path), use_demucs=use_demucs, forced_time_sig=time_signature)
        return JSONResponse(result, headers=NO_CACHE_HEADERS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    launch_url = f"http://127.0.0.1:8000/?v={int(time.time())}"
    print(f"Starting Chord Detector Offline Server at {launch_url} ...")
    threading.Timer(1.0, lambda: webbrowser.open(launch_url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
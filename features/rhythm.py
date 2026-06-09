# features/rhythm.py
import numpy as np
import librosa


def extract_onsets(y, sr, silence_mask=None):
    """
    Находит вокальные onset события только внутри voice regions.
    """
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=True,
        units="frames",
        pre_max=20,
        post_max=20,
        pre_avg=100,
        post_avg=100,
        delta=0.2,
        wait=0
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if silence_mask is not None:
        valid_onsets = []
        for t in onset_times:
            idx = int(t * sr)
            if idx < len(silence_mask) and silence_mask[idx] > 0.5:
                valid_onsets.append(t)
        return np.array(valid_onsets, dtype=np.float32)

    return np.array(onset_times, dtype=np.float32)


def build_rhythm_grid(duration, resolution=0.5):
    """
    Создаёт искусственную сетку времени (метроном).
    """
    return np.arange(0, duration, resolution, dtype=np.float32)


def compute_intervals(onset_times):
    """
    Считает интервалы между onset событиями.
    """
    if len(onset_times) < 2:
        return np.array([], dtype=np.float32)
    return np.diff(onset_times).astype(np.float32)


def compute_micro_timing(onset_times, grid, tolerance=0.02):
    """
    Считает отклонение onset от ближайшей точки сетки.
    """
    results = []
    for t in onset_times:
        nearest = grid[np.argmin(np.abs(grid - t))]
        error = float(t - nearest)
        if abs(error) <= tolerance:
            status = "perfect"
        elif error < 0:
            status = "early"
        else:
            status = "late"
        results.append({
            "time": float(t),
            "nearest_grid": float(nearest),
            "error": error,
            "status": status
        })
    return results


def get_rhythm(audio_data, resolution=0.5):
    """
    Главный extractor: возвращает словарь с признаками ритма.
    """
    onsets = extract_onsets(
        audio_data.waveform,
        audio_data.sample_rate,
        getattr(audio_data, "silence_mask", None)
    )

    intervals = compute_intervals(onsets)
    grid = build_rhythm_grid(audio_data.duration, resolution)
    micro_timing = compute_micro_timing(onsets, grid)

    return {
        "times": onsets.tolist(),
        "intervals": intervals.tolist(),
        "mean_interval": float(np.mean(intervals)) if len(intervals) > 0 else 0.0,
        "std_interval": float(np.std(intervals)) if len(intervals) > 0 else 0.0,
        "jitter": float(np.mean(np.abs(np.diff(intervals)))) if len(intervals) > 1 else 0.0,
        "micro_timing": micro_timing
    }

# features/pitch.py
import numpy as np
import librosa


class AudioData:
    def __init__(self, waveform, sample_rate):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.duration = len(waveform) / sample_rate
        self.samples = len(waveform)
        self.min_amplitude = float(waveform.min())
        self.max_amplitude = float(waveform.max())
        self.mean_amplitude = float(waveform.mean())
        self.std_amplitude = float(waveform.std())


def extract_pitch(y, sr, frame_length=2048, hop_length=256, method="pyin"):
    fmin = 50.0   # чуть ниже человеческого диапазона
    fmax = 1500.0  # расширенный верх

    if method == "pyin":
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
    elif method == "yin":
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
        voiced_flag = np.isfinite(f0)
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length)[0]
        voiced_prob = rms  # абсолютная энергия
    else:
        raise ValueError("method must be 'pyin' or 'yin'")

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0, voiced_flag, voiced_prob


def postprocess(f0, voiced_flag, voiced_prob, threshold=0.05):
    f0 = np.array(f0, dtype=np.float32)

    # voiced определяется по уверенности
    voiced = np.where((voiced_flag) & (voiced_prob >= threshold), 1, 0)

    # правило: если voiced=0 → f0=0
    f0 = np.where(voiced == 1, f0, 0.0)

    # фильтрация диапазона (50–1500 Hz)
    f0 = np.where((f0 >= 50) & (f0 <= 1500), f0, 0.0)

    # сглаживание (медианный фильтр)
    kernel = 3
    f0_smooth = np.copy(f0)
    for i in range(len(f0)):
        left = max(0, i - kernel)
        right = min(len(f0), i + kernel)
        window = f0[left:right]
        f0_smooth[i] = np.median(
            window[window > 0]) if np.any(window > 0) else 0.0

    return f0_smooth, voiced


def get_pitch(audio_data, method="pyin"):
    times, f0_raw, voiced_flag, voiced_prob = extract_pitch(
        audio_data.waveform,
        audio_data.sample_rate,
        method=method
    )
    f0, voiced = postprocess(f0_raw, voiced_flag, voiced_prob)
    return {"times": times.tolist(), "f0": f0.tolist(), "voiced": voiced.tolist()}

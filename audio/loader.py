# audio/loader.py
import os
import numpy as np
import librosa
from dataclasses import dataclass


@dataclass
class AudioData:
    waveform: np.ndarray
    sample_rate: int
    duration: float
    samples: int
    min_amplitude: float
    max_amplitude: float
    mean_amplitude: float
    std_amplitude: float


class AudioLoader:
    def __init__(self, target_sr: int = 22050, min_duration: float = 0.5):
        self.target_sr = target_sr
        self.min_duration = min_duration

    def _check_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")
        if os.path.getsize(file_path) == 0:
            raise FileNotFoundError(f"Файл {file_path} пустой.")

    def _sanitize_waveform(self, y: np.ndarray) -> np.ndarray:
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def load(self, file_path: str) -> AudioData:
        self._check_file(file_path)

        try:
            y, orig_sr = librosa.load(file_path, sr=None, mono=False)
        except Exception as e:
            raise RuntimeError(f"Ошибка открытия файла {file_path}: {e}")

        if y.ndim > 1:
            y = librosa.to_mono(y)

        if orig_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)
        else:
            self.target_sr = orig_sr

        y = self._sanitize_waveform(y)

        duration = len(y) / self.target_sr
        if duration < self.min_duration:
            raise ValueError(f"Слишком короткая запись ({duration:.2f} сек).")

        return AudioData(
            waveform=y,
            sample_rate=self.target_sr,
            duration=duration,
            samples=len(y),
            min_amplitude=float(np.min(y)),
            max_amplitude=float(np.max(y)),
            mean_amplitude=float(np.mean(y)),
            std_amplitude=float(np.std(y)),
        )

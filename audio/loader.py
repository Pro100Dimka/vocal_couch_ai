# audio/loader.py
import os
import numpy as np
import librosa
from dataclasses import dataclass

TARGET_SR = 22050  # можно поменять на 16000 при необходимости
MIN_DURATION_SEC = 0.5


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


def load_audio(file_path: str, sr: int = TARGET_SR) -> AudioData:
    """
    Надёжный загрузчик аудио:
    1. Проверяет существование файла
    2. Загружает аудио
    3. Конвертирует в моно
    4. Ресемплинг до sr
    5. Приводит к float32
    6. Удаляет NaN и Inf
    7. Проверяет минимальную длину
    8. Возвращает AudioData
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    if os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"Файл {file_path} пустой.")

    try:
        y, orig_sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        raise RuntimeError(f"Ошибка открытия файла {file_path}: {e}")

    if y.ndim > 1:
        y = librosa.to_mono(y)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    else:
        sr = orig_sr

    y = np.asarray(y, dtype=np.float32)

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    duration = len(y) / sr
    if duration < MIN_DURATION_SEC:
        raise ValueError(f"Слишком короткая запись ({duration:.2f} сек).")

    return AudioData(
        waveform=y,
        sample_rate=sr,
        duration=duration,
        samples=len(y),
        min_amplitude=np.min(y),
        max_amplitude=np.max(y),
        mean_amplitude=np.mean(y),
        std_amplitude=np.std(y),
    )

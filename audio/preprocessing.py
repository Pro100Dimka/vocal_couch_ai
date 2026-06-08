# audio/preprocessing.py
import numpy as np
import librosa
import scipy.signal


def normalize_audio(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Нормализация амплитуды"""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y * (target_peak / peak)
    return y


def remove_dc_offset(y: np.ndarray) -> np.ndarray:
    """Удаление DC offset (сдвига относительно нуля)"""
    return y - np.mean(y)


def remove_silence(y: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
    """
    Удаление тишины.
    Возвращает только активные сегменты (например вокал).
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    non_silent = np.concatenate([y[start:end] for start, end in intervals])
    return non_silent


def high_pass_filter(y: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """High-pass фильтр для удаления низкочастотного гула"""
    sos = scipy.signal.butter(
        N=4, Wn=cutoff, btype="highpass", fs=sr, output="sos"
    )
    return scipy.signal.sosfilt(sos, y)


def final_normalize(y: np.ndarray) -> np.ndarray:
    """Финальная нормализация, чтобы гарантировать диапазон [-1, 1]"""
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak  # жёсткая нормализация
    return np.clip(y, -1.0, 1.0)  # страховка от выбросов


def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Полный пайплайн обработки"""
    y = normalize_audio(y)
    y = remove_dc_offset(y)
    y = remove_silence(y, sr)
    y = high_pass_filter(y, sr)
    y = final_normalize(y)
    return y


if __name__ == "__main__":
    # пример использования
    y, sr = librosa.load("audio_samples/my_voice.wav", sr=None)

    processed = preprocess_audio(y, sr)

    print("Исходная длина:", len(y) / sr, "сек")
    print("После обработки:", len(processed) / sr, "сек")

    librosa.output.write_wav(
        "audio_samples/my_voice_processed.wav", processed, sr)

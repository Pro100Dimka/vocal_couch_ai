import librosa
import numpy as np

TARGET_SR = 22050


def load_audio(file_path: str, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load audio file and convert to mono waveform.
    Returns:
        y: np.ndarray (normalized audio signal)
        sr: sample rate
    """
    print(f"[INFO] Попытка загрузить файл: {file_path} с sr={sr}")
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        print(f"[OK] Файл загружен. Длина массива: {len(y)}, sr={sr}")
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке: {e}")
        raise RuntimeError(f"Ошибка загрузки файла {file_path}: {e}")

    # базовая статистика до нормализации
    print(
        f"[DEBUG] До нормализации: min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}, std={np.std(y):.6f}")

    # нормализация громкости [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        print(f"[INFO] Нормализация выполнена. max_val={max_val:.6f}")
    else:
        print("[WARN] max_val=0, нормализация не выполнена")

    # статистика после нормализации
    print(
        f"[DEBUG] После нормализации: min={np.min(y):.6f}, max={np.max(y):.6f}, mean={np.mean(y):.6f}, std={np.std(y):.6f}")

    return y, sr

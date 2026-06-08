# tests/test_loader.py
import librosa
from audio.loader import load_audio, TARGET_SR


def main():
    path = "audio_samples/my_voice.wav"
    audio = load_audio(path)

    # Проверка sample rate
    sr_orig = librosa.get_samplerate(path)
    assert audio.sample_rate == TARGET_SR, f"SR mismatch: {audio.sample_rate} != {TARGET_SR}"
    print(f"[INFO] Original SR: {sr_orig}")
    print(f"[INFO] Sample rate: {audio.sample_rate}")

    # Проверка количества сэмплов
    expected_samples = int(audio.duration * audio.sample_rate)
    assert audio.samples == expected_samples, f"Samples mismatch: {audio.samples} != {expected_samples}"

    # Логирование статистики
    print(f"[INFO] Файл: {path}")
    print(f"[INFO] Длительность: {audio.duration:.2f} сек")
    print(f"[INFO] Сэмплов: {audio.samples}")
    print("[INFO] Channels: 1")
    print(f"[INFO] Min amplitude: {audio.min_amplitude}")
    print(f"[INFO] Max amplitude: {audio.max_amplitude}")
    print(f"[INFO] Mean amplitude: {audio.mean_amplitude}")
    print(f"[INFO] Std amplitude: {audio.std_amplitude}")
    print(f"[INFO] Waveform dtype: {audio.waveform.dtype}")

    print("[TEST] Все проверки пройдены успешно!")


if __name__ == "__main__":
    main()

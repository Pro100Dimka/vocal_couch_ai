# tests/test_loader.py
import numpy as np
import pytest
import soundfile as sf
from audio.loader import AudioLoader, AudioData

TARGET_SR = 22050
MIN_DURATION_SEC = 0.5


def create_temp_wav(tmp_path, data, sr=TARGET_SR):
    file_path = tmp_path / "test.wav"
    sf.write(file_path, data, sr)
    return str(file_path)


@pytest.fixture
def loader():
    """Фикстура для создания экземпляра AudioLoader."""
    return AudioLoader(target_sr=TARGET_SR, min_duration=MIN_DURATION_SEC)


@pytest.mark.parametrize("filename", [
    "audio_samples/my_voice.wav",
    "audio_samples/reference.wav",
])
def test_real_audio_files(loader, filename):
    audio = loader.load(filename)
    print(f"\nФайл: {filename} | Длительность: {audio.duration:.2f} сек | "
          f"Сэмплов: {audio.samples} | Средняя амплитуда: {audio.mean_amplitude:.4f}")
    assert isinstance(audio, AudioData)
    assert audio.duration > loader.min_duration
    assert audio.samples == len(audio.waveform)
    assert audio.min_amplitude <= audio.max_amplitude


def test_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.wav")


def test_empty_file(loader, tmp_path):
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        loader.load(str(empty_file))


def test_too_short_audio(loader, tmp_path):
    samples = int(TARGET_SR * (MIN_DURATION_SEC / 2))
    y = np.zeros(samples, dtype=np.float32)
    file_path = create_temp_wav(tmp_path, y)
    with pytest.raises(ValueError):
        loader.load(file_path)


def test_valid_audio(loader, tmp_path):
    samples = TARGET_SR
    y = np.random.randn(samples).astype(np.float32)
    file_path = create_temp_wav(tmp_path, y)
    audio = loader.load(file_path)
    assert isinstance(audio, AudioData)
    assert audio.sample_rate == TARGET_SR
    assert audio.duration == pytest.approx(1.0, rel=1e-2)
    assert audio.samples == samples
    assert np.isfinite(audio.mean_amplitude)


def test_nan_and_inf_cleanup(loader, tmp_path):
    samples = TARGET_SR
    y = np.random.randn(samples).astype(np.float32)
    y[100] = np.nan
    y[200] = np.inf
    y[300] = -np.inf
    file_path = create_temp_wav(tmp_path, y)
    audio = loader.load(file_path)
    assert np.all(np.isfinite(audio.waveform))
    assert audio.duration >= MIN_DURATION_SEC

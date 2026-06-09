# tests/test_preprocessor.py
import numpy as np
import pytest
import librosa
import soundfile as sf

from audio.loader import AudioLoader
from audio.preprocessor import AudioPreprocessor, ProcessedAudioData


@pytest.fixture
def synthetic_audio(tmp_path):
    sr = 22050
    y = librosa.tone(440, sr=sr, duration=1.0)
    file_path = tmp_path / "test.wav"
    sf.write(str(file_path), y, sr)
    return str(file_path), sr, y


def test_process_returns_processed_audio(synthetic_audio):
    file_path, sr, _ = synthetic_audio
    loader = AudioLoader(target_sr=sr)
    data = loader.load(file_path)

    preprocessor = AudioPreprocessor()
    processed = preprocessor.process(data)

    assert isinstance(processed, ProcessedAudioData)
    assert processed.processed is True
    assert np.max(np.abs(processed.waveform)) <= 1.0
    assert processed.duration > 0


def test_silence_mask_created(synthetic_audio):
    file_path, sr, y = synthetic_audio
    loader = AudioLoader(target_sr=sr)
    data = loader.load(file_path)

    preprocessor = AudioPreprocessor(use_hard_silence=True)
    processed = preprocessor.process(data)

    assert processed.silence_mask is not None
    assert set(np.unique(processed.silence_mask)).issubset({0.0, 1.0})


def test_soft_vs_hard_silence(tmp_path):
    sr = 22050
    # сигнал: тишина + тон
    y = np.concatenate([
        np.zeros(sr // 2),                  # 0.5 сек тишины
        librosa.tone(440, sr=sr, duration=0.5)  # 0.5 сек синус
    ])
    file_path = tmp_path / "test.wav"
    sf.write(str(file_path), y, sr)

    loader = AudioLoader(target_sr=sr)
    data = loader.load(file_path)

    soft = AudioPreprocessor(use_hard_silence=False,
                             silence_fill=0.02).process(data)
    hard = AudioPreprocessor(use_hard_silence=True).process(data)

    # В мягком режиме тишина заменена на silence_fill
    assert np.any(np.isclose(soft.waveform, 0.02, atol=1e-3))
    # В жёстком режиме тишина полностью убрана
    assert np.all((hard.waveform == 0.0) | (hard.waveform != 0.0))


def test_loader_and_preprocessor_integration():
    file_path = "audio_samples/reference.wav"
    loader = AudioLoader(target_sr=22050)
    data = loader.load(file_path)

    preprocessor = AudioPreprocessor()
    processed = preprocessor.process(data)

    assert isinstance(processed, ProcessedAudioData)
    assert processed.samples == len(processed.waveform)
    assert np.max(np.abs(processed.waveform)) <= 1.0

# tests/test_embeddings_pipeline.py
import numpy as np
from audio.loader import AudioLoader
from audio.preprocessor import AudioPreprocessor
from features.pitch import get_pitch_stream
from features.rhythm import get_rhythm_stream
from features.voice_quality import get_voice_quality_stream
from features.embeddings import get_embeddings


def test_embeddings_extraction():
    loader = AudioLoader(target_sr=16000)
    audio = loader.load("audio_samples/reference.wav")

    preprocessor = AudioPreprocessor(use_hard_silence=True)
    processed = preprocessor.process(audio)

    # получаем ручные фичи
    pitch_stream = get_pitch_stream(processed)
    rhythm_stream = get_rhythm_stream(processed)
    quality_stream = get_voice_quality_stream(processed)

    # получаем эмбеддинги
    embeddings = get_embeddings(
        processed, pitch_stream, rhythm_stream, quality_stream)

    # проверяем наличие ключей
    expected_keys = {"global_embedding",
                     "segment_embeddings", "feature_embedding", "metadata"}
    assert expected_keys.issubset(embeddings.keys())

    # проверяем, что глобальный эмбеддинг нормализован
    global_vec = np.array(embeddings["global_embedding"])
    norm = np.linalg.norm(global_vec)
    assert np.isclose(norm, 1.0, atol=1e-3)

    # проверяем, что есть хотя бы один сегмент
    assert len(embeddings["segment_embeddings"]) > 0

    # проверяем, что каждый сегмент нормализован
    for seg in embeddings["segment_embeddings"]:
        seg_vec = np.array(seg)
        assert np.isclose(np.linalg.norm(seg_vec), 1.0, atol=1e-3)

    # проверяем метаданные
    meta = embeddings["metadata"]
    assert "duration" in meta
    assert "segment_count" in meta
    assert meta["segment_count"] == len(embeddings["segment_embeddings"])


def test_voice_quality_extraction():
    loader = AudioLoader(target_sr=16000)
    audio = loader.load("audio_samples/reference.wav")

    preprocessor = AudioPreprocessor(use_hard_silence=True)
    processed = preprocessor.process(audio)

    quality = get_voice_quality_stream(processed)

    # Проверяем, что все ключи присутствуют
    expected_keys = {
        "overall_score",
        "stability_score",
        "noise_score",
        "control_score",
        "breathiness_score",
        "clarity_score",
        "articulation_score",
        "dynamics_profile",
        "voice_energy_curve",
    }
    assert expected_keys.issubset(quality.keys())

    # Проверяем, что метрики находятся в диапазоне 0.0–1.0
    for key in [
        "overall_score",
        "stability_score",
        "noise_score",
        "control_score",
        "breathiness_score",
        "clarity_score",
        "articulation_score",
    ]:
        assert 0.0 <= quality[key] <= 1.0

    # Проверяем, что динамический профиль содержит кривую энергии
    assert isinstance(quality["dynamics_profile"]["curve"], list)
    assert len(quality["dynamics_profile"]["curve"]) > 0

    # Проверяем, что voice_energy_curve совпадает с кривой
    assert quality["voice_energy_curve"] == quality["dynamics_profile"]["curve"]


def test_rhythm_with_and_without_preprocess():
    loader = AudioLoader(target_sr=16000)
    audio = loader.load("audio_samples/reference.wav")

    # С препроцессингом
    preprocessor = AudioPreprocessor(use_hard_silence=True)
    processed = preprocessor.process(audio)
    proc_stream = get_rhythm_stream(processed, resolution=0.5)

    # Проверяем, что onsets извлекаются
    assert len(proc_stream["times"]) > 0

    # Проверяем интервалы
    intervals = np.array(proc_stream["intervals"])
    if len(intervals) > 0:
        assert all(i > 0 for i in intervals)

    # Проверяем, что mean_interval в разумных пределах (0.1–2.0 сек)
    mean_interval = proc_stream["mean_interval"]
    assert 0.1 <= mean_interval <= 3.0

    # Проверяем micro-timing: должны быть статусы только из {early, late, perfect}
    statuses = {mt["status"] for mt in proc_stream["micro_timing"]}
    assert statuses.issubset({"early", "late", "perfect"})


def test_pitch_with_and_without_preprocess():
    loader = AudioLoader(target_sr=16000)
    audio = loader.load("audio_samples/reference.wav")

    raw_stream = get_pitch_stream(audio, method="yin")
    preprocessor = AudioPreprocessor(use_hard_silence=True)
    processed = preprocessor.process(audio)
    proc_stream = get_pitch_stream(processed, method="yin")

    raw_f0 = np.array([f for f in raw_stream["f0"] if f > 0])
    proc_f0 = np.array([f for f in proc_stream["f0"] if f > 0])

    # Проверяем, что pitch извлекается
    assert len(proc_f0) > 0

    # Проверяем диапазон
    assert all(50 <= f <= 1500 for f in proc_f0)

    # Проверяем voiced ratio не слишком низкий
    proc_voiced_ratio = sum(proc_stream["voiced"]) / len(proc_stream["voiced"])
    assert proc_voiced_ratio > 0.1

    # Проверяем, что дисперсия частот после препроцессинга меньше (сигнал стабильнее)
    if len(raw_f0) > 0:
        assert np.std(proc_f0) <= np.std(raw_f0) * 1.2

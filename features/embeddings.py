# features/embeddings.py
import numpy as np
import librosa
from sklearn.preprocessing import normalize


def frame_audio(y, sr, frame_size=3.0, hop_size=1.5):
    """Разбивает сигнал на окна (3 сек, шаг 1.5 сек)."""
    frame_len = int(frame_size * sr)
    hop_len = int(hop_size * sr)
    frames = []
    for start in range(0, len(y) - frame_len, hop_len):
        frames.append(y[start:start + frame_len])
    return frames


def compute_embedding(y, sr):
    """
    Базовый embedding через спектральные признаки.
    В реальной системе заменяется на CNN/Transformer encoder (wav2vec2/HuBERT).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    vec = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(spec_centroid, axis=1),
        np.mean(spec_bandwidth, axis=1)
    ])
    return vec


def silence_aware_pooling(embeddings, silence_mask=None):
    """Агрегация эмбеддингов с учётом маски тишины."""
    if len(embeddings) == 0:
        return np.zeros(10)
    if silence_mask is None or len(silence_mask) == 0:
        pooled = np.mean(embeddings, axis=0)
    else:
        weights = np.linspace(0.1, 1.0, len(embeddings))
        pooled = np.average(embeddings, axis=0, weights=weights)
    return pooled


def fuse_features(pitch_stream=None, rhythm_stream=None, quality_stream=None):
    """Фьюжн ручных фич в отдельный вектор."""
    fused = []
    if pitch_stream is not None:
        fused.append(np.mean(pitch_stream.get("f0", [0.0])))
    if rhythm_stream is not None:
        fused.append(rhythm_stream.get("mean_interval", 0.0))
        fused.append(rhythm_stream.get("std_interval", 0.0))
    if quality_stream is not None:
        fused.append(quality_stream.get("overall_score", 0.0))
        fused.append(quality_stream.get("clarity_score", 0.0))
    return np.array(fused, dtype=np.float32)


def get_embeddings(audio_data, pitch_stream=None, rhythm_stream=None, quality_stream=None):
    """
    Главный extractor: возвращает глобальный и сегментные эмбеддинги.
    """
    y = audio_data.waveform
    sr = audio_data.sample_rate

    # сегментные эмбеддинги
    frames = frame_audio(y, sr)
    segment_embeddings = [compute_embedding(f, sr) for f in frames]

    # глобальный эмбеддинг (silence-aware pooling)
    global_embedding = silence_aware_pooling(
        segment_embeddings, getattr(audio_data, "silence_mask", None))

    # фьюжн ручных фич
    feature_embedding = fuse_features(
        pitch_stream, rhythm_stream, quality_stream)

    # нормализация
    global_embedding = normalize(global_embedding.reshape(1, -1))[0]
    segment_embeddings = [normalize(seg.reshape(1, -1))[0]
                          for seg in segment_embeddings]
    feature_embedding = normalize(feature_embedding.reshape(
        1, -1))[0] if feature_embedding.size > 0 else None

    return {
        "global_embedding": global_embedding.tolist(),
        "segment_embeddings": [seg.tolist() for seg in segment_embeddings],
        "feature_embedding": feature_embedding.tolist() if feature_embedding is not None else None,
        "metadata": {
            "duration": audio_data.duration,
            "segment_count": len(segment_embeddings),
            "silence_ratio": float(np.mean(audio_data.silence_mask)) if hasattr(audio_data, "silence_mask") else None,
            # простая эвристика надёжности
            "confidence": float(1.0 - np.std(global_embedding))
        }
    }

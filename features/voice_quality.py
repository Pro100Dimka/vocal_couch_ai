# features/voice_quality.py
import numpy as np
import librosa


def compute_stability(y):
    """Стабильность амплитуды: чем меньше std, тем выше score."""
    return float(1.0 / (1.0 + np.std(y)))


def compute_noise(y, sr):
    """Шум: spectral flatness (0 = тон, 1 = шум)."""
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    return float(1.0 - np.mean(flatness))  # ближе к 1 → чище


def compute_dynamics(y, sr, silence_mask=None):
    """Контроль громкости: RMS кривая с мелким шагом и маской."""
    hop_length = 128
    frame_length = 1024
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length)[0]

    if silence_mask is not None:
        # переводим маску в фреймы той же длины, что и RMS
        mask_frames = librosa.util.frame(
            silence_mask.astype(np.float32),
            frame_length=frame_length,
            hop_length=hop_length
        ).mean(axis=0)

        # подгоняем длину маски к RMS
        min_len = min(len(rms), len(mask_frames))
        rms = rms[:min_len] * mask_frames[:min_len]

    dynamics_range = float(np.max(rms) - np.min(rms))
    score = float(1.0 / (1.0 + dynamics_range))
    return score, rms.tolist()


def compute_hnr(y, sr):
    """Harmonics-to-noise ratio (через cepstrum)."""
    S = np.abs(librosa.stft(y))
    autocorr = librosa.autocorrelate(np.mean(S, axis=1))
    hnr = np.max(autocorr[1:]) / (autocorr[0] + 1e-6)
    return float(hnr / (hnr + 1.0))  # нормализуем в [0,1]


def compute_spectral_tilt(y, sr):
    """Spectral tilt: наклон спектра (воздушность)."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    energy = np.mean(S, axis=1)
    coeffs = np.polyfit(freqs, energy, 1)
    tilt = coeffs[0]
    return float(1.0 / (1.0 + abs(tilt) / 1000.0))


def compute_breathiness(y, sr):
    """Дыхательность: комбинация HNR и spectral tilt."""
    hnr_score = compute_hnr(y, sr)
    tilt_score = compute_spectral_tilt(y, sr)
    return float((1.0 - hnr_score + tilt_score) / 2.0)


def compute_clarity(y, sr):
    """Ясность: энергия в голосовом диапазоне."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= 80) & (freqs <= 3000)
    band_energy = np.sum(S[mask, :])
    total_energy = np.sum(S)
    return float(band_energy / (total_energy + 1e-6))


def compute_articulation(y, sr, silence_mask):
    """Атака: резкие переходы silence→voice."""
    if silence_mask is None:
        return 0.5
    diffs = np.diff(silence_mask.astype(np.float32))
    attacks = np.sum(diffs == 1.0)
    return float(min(1.0, attacks / (len(y) / sr)))


def get_voice_quality(audio_data):
    """
    Главный extractor: возвращает словарь с признаками качества голоса.
    """
    y = audio_data.waveform
    sr = audio_data.sample_rate

    stability_score = compute_stability(y)
    noise_score = compute_noise(y, sr)
    control_score, energy_curve = compute_dynamics(
        y, sr, getattr(audio_data, "silence_mask", None))
    breathiness_score = compute_breathiness(y, sr)
    clarity_score = compute_clarity(y, sr)
    articulation_score = compute_articulation(
        y, sr, getattr(audio_data, "silence_mask", None))

    overall_score = float(np.mean([
        stability_score,
        noise_score,
        control_score,
        breathiness_score,
        clarity_score,
        articulation_score
    ]))

    return {
        "overall_score": overall_score,
        "stability_score": stability_score,
        "noise_score": noise_score,
        "control_score": control_score,
        "breathiness_score": breathiness_score,
        "clarity_score": clarity_score,
        "articulation_score": articulation_score,
        "dynamics_profile": {
            "range": float(np.max(energy_curve) - np.min(energy_curve)),
            "curve": energy_curve
        },
        "voice_energy_curve": energy_curve
    }

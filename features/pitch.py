import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


# =========================
# ENERGY MASK
# =========================
def energy_mask(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 256,
    threshold: float | None = None
) -> np.ndarray:

    print(f"[energy_mask] start | samples={len(y)}")

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    print(
        f"[energy_mask] RMS | len={len(rms)} "
        f"min={np.min(rms):.6f} max={np.max(rms):.6f} mean={np.mean(rms):.6f}"
    )

    if threshold is None:
        threshold = np.percentile(rms, 10)
        print(f"[energy_mask] adaptive threshold=10% => {threshold:.8f}")
    else:
        print(f"[energy_mask] fixed threshold={threshold:.8f}")

    mask = rms > threshold

    print(
        f"[energy_mask] voiced_frames={np.sum(mask)} / {len(mask)} "
        f"({100*np.mean(mask):.2f}%)"
    )

    return mask


# =========================
# PITCH EXTRACTION
# =========================
def extract_pitch(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 256,
    method: str = "pyin"
):

    print(f"\n[extract_pitch] START | sr={sr} method={method}")
    print(
        f"[extract_pitch] audio samples={len(y)} duration≈{len(y)/sr:.2f}s")

    fmin = librosa.note_to_hz("E2")
    fmax = librosa.note_to_hz("B5")

    print(f"[extract_pitch] fmin={fmin:.2f} fmax={fmax:.2f}")

    try:
        if method == "pyin":
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                sr=sr,
                fmin=fmin,
                fmax=fmax,
                frame_length=frame_length,
                hop_length=hop_length
            )

        elif method == "yin":
            f0 = librosa.yin(
                y,
                sr=sr,
                fmin=fmin,
                fmax=fmax,
                frame_length=frame_length,
                hop_length=hop_length
            )
            voiced_flag = np.isfinite(f0)
            voiced_prob = None

        else:
            raise ValueError("method must be 'yin' or 'pyin'")

    except Exception as e:
        logger.exception(f"[extract_pitch] FAILED: {e}")
        raise

    f0 = np.array(f0)

    # =========================
    # DIAGNOSTICS
    # =========================
    valid = f0[np.isfinite(f0)]

    print(
        f"[extract_pitch] f0 stats | "
        f"len={len(f0)} "
        f"NaN={np.isnan(f0).sum()} "
        f"valid={len(valid)}"
    )

    if len(valid) > 0:
        print(
            f"[extract_pitch] pitch range | "
            f"min={np.min(valid):.2f} max={np.max(valid):.2f}"
        )

        print(
            f"[extract_pitch] percentiles | "
            f"P5={np.percentile(valid, 5):.2f} "
            f"P50={np.percentile(valid, 50):.2f} "
            f"P95={np.percentile(valid, 95):.2f}"
        )

    # 🔥 DETECT "ZALIPANIE NA fmax"
    near_fmax = np.sum(f0 >= fmax * 0.99)
    if near_fmax > 0:
        logger.warning(
            f"[extract_pitch] ⚠ fmax sticking detected: "
            f"{near_fmax}/{len(f0)} ({100*near_fmax/len(f0):.2f}%)"
        )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    print(f"[extract_pitch] times={len(times)}")

    return times, f0, voiced_flag, voiced_prob


# =========================
# SMOOTHING
# =========================
def smooth_pitch(f0: np.ndarray, win: int = 5) -> np.ndarray:

    print(f"[smooth_pitch] start win={win}")

    f0 = np.array(f0, dtype=np.float32)
    out = np.copy(f0)

    replaced = 0

    for i in range(len(f0)):
        start = max(0, i - win)
        end = min(len(f0), i + win + 1)

        window = f0[start:end]
        valid = window[np.isfinite(window)]

        if len(valid) >= win // 2:
            out[i] = np.median(valid)
            replaced += 1

    print(f"[smooth_pitch] replaced={replaced}/{len(f0)}")

    return out


# =========================
# HZ -> MIDI
# =========================
def hz_to_midi(f0: np.ndarray) -> np.ndarray:

    print("[hz_to_midi] converting")

    f0 = np.array(f0, dtype=np.float32)

    invalid = np.sum((f0 < 20) | (f0 > 5000) | np.isnan(f0))
    print(f"[hz_to_midi] invalid={invalid}/{len(f0)}")

    f0 = np.where((f0 > 20) & (f0 < 5000), f0, np.nan)

    midi = 69 + 12 * np.log2(f0 / 440.0)

    print(f"[hz_to_midi] NaN midi={np.isnan(midi).sum()}")

    return midi


# =========================
# MAIN PIPELINE
# =========================
def get_pitch_features(
    y: np.ndarray,
    sr: int,
    method: str = "pyin"
):

    print("\n================ PIPELINE START ================")

    times, f0_raw, voiced_flag, voiced_prob = extract_pitch(
        y, sr, method=method
    )

    print(
        f"[pipeline] raw f0 | min={np.nanmin(f0_raw):.2f} "
        f"max={np.nanmax(f0_raw):.2f}"
    )

    f0 = smooth_pitch(f0_raw)
    midi = hz_to_midi(f0)

    energy_voiced = energy_mask(y)

    min_len = min(len(f0), len(energy_voiced), len(voiced_flag))

    print(
        f"[pipeline] alignment | "
        f"f0={len(f0)} energy={len(energy_voiced)} voiced={len(voiced_flag)} min={min_len}"
    )

    voiced = voiced_flag[:min_len] | energy_voiced[:min_len]

    print(
        f"[pipeline] FINAL voiced ratio={np.mean(voiced):.3f}"
    )

    print("================ PIPELINE END ================\n")

    return {
        "times": times[:min_len],
        "f0_raw": f0_raw[:min_len],
        "f0": f0[:min_len],
        "midi": midi[:min_len],
        "voiced": voiced,
        "voiced_prob": voiced_prob[:min_len] if voiced_prob is not None else None
    }

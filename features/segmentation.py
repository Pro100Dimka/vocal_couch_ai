# features/segmentation.py
import numpy as np
from analysis.pitch_analyzer import hz_to_midi, midi_to_note_name, merge_short_notes, smooth_notes


def smooth_pitch(f0, window=5):
    f0 = np.array(f0, dtype=np.float32)
    return np.convolve(f0, np.ones(window)/window, mode="same").tolist()


def detect_note_segments(f0, voiced, times, min_duration=0.08):
    segments = []
    start, pitches = None, []
    for i in range(len(f0)):
        if voiced[i] and f0[i] > 0:
            if start is None:
                start = times[i]
                pitches = []
            pitches.append(f0[i])
        else:
            if start is not None and len(pitches) > 0:
                end = times[i]
                duration = end - start
                if duration >= min_duration:
                    pitch_mean = float(np.mean(pitches))
                    pitch_std = float(np.std(pitches))
                    stability = float(1.0 / (1.0 + pitch_std))
                    voiced_ratio = float(
                        len(pitches) / (i - int(start * len(f0) / times[-1])))

                    midi_note = int(hz_to_midi(pitch_mean))
                    note = midi_to_note_name(midi_note)
                    pitch_confidence = float(
                        1.0 - (pitch_std / (pitch_mean + 1e-6)))
                    energy = float(np.mean(np.abs(pitches)) /
                                   (np.max(pitches) + 1e-6))
                    attack = float(
                        (pitches[1] - pitches[0]) / (pitch_mean + 1e-6)) if len(pitches) > 1 else 0.0
                    decay = float(
                        (pitches[-1] - pitches[-2]) / (pitch_mean + 1e-6)) if len(pitches) > 2 else 0.0

                    # deviation in cents
                    deviation_cents = float(
                        1200 * np.log2(pitch_mean / (440.0 * 2 ** ((midi_note - 69) / 12))))

                    # semantic quality
                    if stability > 0.85 and pitch_confidence > 0.8:
                        quality = "stable"
                    elif pitch_confidence < 0.5:
                        quality = "shaky"
                    elif energy < 0.3:
                        quality = "breathy"
                    else:
                        quality = "noisy"

                    control_score = float(
                        (stability + pitch_confidence + energy) / 3.0)

                    segments.append({
                        # базовые музыкальные
                        "midi": midi_note,
                        "note": note,
                        "freq": pitch_mean,              # то же самое, что f0_mean
                        "f0_mean": pitch_mean,           # явное поле для совместимости
                        "start": start,
                        "end": end,
                        "duration": duration,

                        # стабильность
                        "stability": stability,
                        "f0_stability": stability,       # алиас для совместимости
                        "voiced_ratio": voiced_ratio,

                        # дополнительные музыкальные признаки
                        "pitch_confidence": pitch_confidence,
                        "energy": energy,
                        "attack": attack,
                        "decay": decay,
                        "deviation_cents": deviation_cents,

                        # семантика
                        "quality": quality,
                        "control_score": control_score,
                        "type": "sung_note"
                    })
                start, pitches = None, []
    return segments


def detect_phrases(note_segments, pause_threshold=0.5):
    phrases = []
    if not note_segments:
        return phrases
    current_phrase = {"start": note_segments[0]["start"], "notes": []}
    for i, seg in enumerate(note_segments):
        current_phrase["notes"].append(seg)
        if i < len(note_segments) - 1:
            gap = note_segments[i+1]["start"] - seg["end"]
            if gap > pause_threshold:
                current_phrase["end"] = seg["end"]
                current_phrase["note_count"] = len(current_phrase["notes"])
                current_phrase["avg_stability"] = float(
                    np.mean([n["stability"] for n in current_phrase["notes"]])
                )
                phrases.append(current_phrase)
                current_phrase = {
                    "start": note_segments[i+1]["start"], "notes": []}
    # завершаем последнюю фразу
    current_phrase["end"] = note_segments[-1]["end"]
    current_phrase["note_count"] = len(current_phrase["notes"])
    current_phrase["avg_stability"] = float(
        np.mean([n["stability"] for n in current_phrase["notes"]])
    )
    phrases.append(current_phrase)
    return phrases


def compute_timing_structure(phrases, total_duration):
    """Анализ тайминговой структуры."""
    total_phrases = len(phrases)
    avg_phrase_length = float(
        np.mean([p["end"] - p["start"] for p in phrases])) if phrases else 0.0
    pause_ratio = float(
        sum([phrases[i+1]["start"] - phrases[i]["end"]
            for i in range(len(phrases)-1)]) / total_duration
    ) if total_phrases > 1 else 0.0
    return {
        "total_phrases": total_phrases,
        "avg_phrase_length": avg_phrase_length,
        "pause_ratio": pause_ratio
    }


def get_segmentation_features(pitch_stream):
    """
    Главный extractor: превращает pitch stream в ноты, фразы и тайминговую структуру.
    """
    f0 = pitch_stream["f0"]
    voiced = pitch_stream["voiced"]
    times = pitch_stream["times"]

    # шаг 1 — сглаживание pitch contour
    f0_smooth = smooth_pitch(f0)

    # шаг 2 — извлечение нотных сегментов
    note_segments = detect_note_segments(f0_smooth, voiced, times)

    # шаг 3 — пост‑обработка сегментов
    note_segments = merge_short_notes(note_segments, min_duration=0.08)
    note_segments = smooth_notes(note_segments)

    # шаг 4 — группировка в фразы
    phrases = detect_phrases(note_segments)

    # шаг 5 — тайминговая структура
    timing = compute_timing_structure(phrases, times[-1])

    # шаг 6 — глобальная мета‑статистика
    metadata = {
        "total_notes": len(note_segments),
        "avg_energy": float(np.mean([n["energy"] for n in note_segments])) if note_segments else 0.0,
        "avg_stability": float(np.mean([n["stability"] for n in note_segments])) if note_segments else 0.0
    }

    return {
        "note_segments": note_segments,
        "phrase_segments": phrases,
        "timing_structure": timing,
        "metadata": metadata
    }

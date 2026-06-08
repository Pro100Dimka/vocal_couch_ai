# analysis/pitch_analyzer.py
import numpy as np
from typing import List, Dict, Any

# ---------------- Вспомогательные функции ----------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_note_name(midi: float) -> str:
    midi_int = int(round(midi))
    octave = (midi_int // 12) - 1
    name = NOTE_NAMES[midi_int % 12]
    return f"{name}{octave}"


def merge_short_notes(notes: List[Dict[str, Any]], min_duration: float = 0.08) -> List[Dict[str, Any]]:
    merged = []
    for n in notes:
        if merged and n["note"] == merged[-1]["note"] and (n["start"] - merged[-1]["end"]) < 0.08:
            merged[-1]["end"] = n["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            if n["duration"] >= min_duration:
                merged.append(n)
    return merged


def smooth_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    smoothed = []
    for i, n in enumerate(notes):
        current_midi = int(round(hz_to_midi(n["freq"])))
        if i > 0:
            prev = smoothed[-1]
            if abs(prev.get("midi", current_midi) - current_midi) <= 1 and abs(n.get("deviation_cents", 0)) < 40:
                # оставляем предыдущую ноту
                continue
        n["midi"] = current_midi
        smoothed.append(n)
    return smoothed


# ---------------- Базовый анализ ----------------


def filter_segments(segments: List[Dict[str, Any]], min_duration: float, stability_threshold: float) -> List[Dict[str, Any]]:
    return [
        seg for seg in segments
        if seg["duration"] >= min_duration and seg["stability"] >= stability_threshold
    ]


def merge_segments(segments: List[Dict[str, Any]], gap_threshold: float = 0.1) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["note"] == last["note"] and seg["start"] - last["end"] <= gap_threshold:
            last["end"] = seg["end"]
            last["duration"] = last["end"] - last["start"]
            last["stability"] = (last["stability"] + seg["stability"]) / 2
        else:
            merged.append(seg)
    return merged


def analyze_pitch_stability(notes: List[Dict[str, Any]]) -> float:
    return float(np.mean([n["stability"] for n in notes])) if notes else 0.0


def analyze_rhythm(notes: List[Dict[str, Any]]) -> float:
    if len(notes) < 2:
        return 0.0
    total_time = notes[-1]["end"] - notes[0]["start"]
    return len(notes) / total_time if total_time > 0 else 0.0


def analyze_intervals(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(notes) < 2:
        return {"avg_interval_jump": 0.0, "large_jumps_count": 0, "melodic_range_semitones": 0}
    midi_vals = [hz_to_midi(n["freq"]) for n in notes]
    jumps = [abs(midi_vals[i] - midi_vals[i-1])
             for i in range(1, len(midi_vals))]
    return {
        "avg_interval_jump": float(np.mean(jumps)),
        "large_jumps_count": sum(1 for j in jumps if j >= 5),
        "melodic_range_semitones": int(round(max(midi_vals) - min(midi_vals)))
    }


def detect_issues(notes: List[Dict[str, Any]], stats: Dict[str, Any], intervals: Dict[str, Any]) -> List[str]:
    issues = []
    if stats["avg_stability"] < 0.9:
        issues.append("slight instability in long notes")
    if intervals["avg_interval_jump"] > 4:
        issues.append("sharp transitions between registers")
    if stats["avg_note_duration"] < 0.2:
        issues.append("notes held too short")
    return issues


def compute_score(stats: Dict[str, Any], intervals: Dict[str, Any]) -> Dict[str, Any]:
    stability = stats["avg_stability"]
    rhythm = 1.0 - abs(stats["avg_note_duration"] - 0.5)
    smoothness = 1.0 - (intervals["avg_interval_jump"] / 12.0)
    score = stability * 40 + rhythm * 20 + smoothness * \
        20 + (1 - stats["break_count"]/10) * 20
    score = max(0, min(100, score))
    if score > 80:
        interp, level = "stable and controlled singing", "good"
    elif score > 60:
        interp, level = "stable but slightly jumpy transitions", "average"
    else:
        interp, level = "unstable and fragmented", "poor"
    return {"score": int(score), "level": level, "interpretation": interp}


# ---------------- Расширенный анализ ----------------
def compute_pitch_deviation(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deviations = []
    for n in notes:
        midi = hz_to_midi(n["freq"])
        target_freq = 440.0 * (2 ** ((round(midi) - 69) / 12))
        cents = 1200 * np.log2(n["freq"] / target_freq)
        deviations.append({
            **n,
            "midi": int(round(midi)),
            "pitch_confidence": n.get("stability", 0.9),
            "deviation_cents": cents
        })
    return deviations


def analyze_vibrato(f0: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
    if f0 is None or len(f0) < 2:
        return {"frequency": 0.0, "depth_cents": 0.0}
    detrended = f0 - np.mean(f0)
    fft = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(len(detrended), d=(times[1]-times[0]))
    peak_idx = np.argmax(np.abs(fft[1:])) + 1
    vib_freq = freqs[peak_idx]
    vib_depth = 1200 * np.log2(1 + np.std(detrended)/np.mean(f0))
    # ограничиваем до реалистичных значений
    vib_depth = max(20.0, min(vib_depth, 150.0))
    return {"frequency": vib_freq, "depth_cents": vib_depth}


def compute_overall_score(pitch_accuracy: float, stability: float, timing: float) -> Dict[str, Any]:
    score = pitch_accuracy * 0.6 + stability * 0.2 + timing * 0.2
    return {"overall_score": round(score * 100, 1)}


def analyze_timing(notes: List[Dict[str, Any]], tempo: float = 120.0) -> Dict[str, Any]:
    if not notes:
        return {"tempo": tempo, "avg_attack_offset": 0.0, "avg_duration_ratio": 0.0}
    beat_duration = 60.0 / tempo
    attack_offsets = []
    duration_ratios = []
    for n in notes:
        expected_beats = round(n["duration"] / beat_duration)
        duration_ratios.append(
            n["duration"] / (expected_beats * beat_duration if expected_beats else beat_duration))
        attack_offsets.append((n["start"] % beat_duration) / beat_duration)
    return {
        "tempo": tempo,
        "avg_attack_offset": float(np.mean(attack_offsets)),
        "avg_duration_ratio": float(np.mean(duration_ratios))
    }


def compare_to_reference(notes: List[Dict[str, Any]], reference: List[str]) -> Dict[str, Any]:
    if not notes or not reference:
        return {"match_rate": 0.0, "match_rate_percent": 0.0}
    sung_notes = [n["note"] for n in notes]
    matches = sum(1 for n in sung_notes if n in reference)
    match_rate = matches / max(len(reference), len(sung_notes))
    return {
        "match_rate": round(match_rate, 2),
        "match_rate_percent": round(match_rate * 100, 1)
    }


# ---------------- Главные классы ----------------
class PitchAnalyzer:
    def __init__(self, stability_threshold: float = 0.85, min_duration: float = 0.08):
        self.stability_threshold = stability_threshold
        self.min_duration = min_duration

    def analyze(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        filtered = filter_segments(
            segments, self.min_duration, self.stability_threshold)
        merged = merge_segments(filtered)
        stats = {
            "avg_stability": np.mean([
                compute_stability(np.array([n["freq"]])) for n in merged
            ]) if merged else 0.0,
            "avg_note_duration": float(np.mean([n["duration"] for n in merged])) if merged else 0.0,
            "note_changes_per_second": analyze_rhythm(merged),
            "break_count": sum(1 for s in segments if s.get("type") == "break")
        }
        intervals = analyze_intervals(merged)
        issues = detect_issues(merged, stats, intervals)
        pitch_quality = compute_score(stats, intervals)
        return {
            "notes": merged,
            "statistics": stats,
            "musical_structure": intervals,
            "issues": issues,
            "pitch_quality": pitch_quality
        }


def compute_intonation_score(deviation_cents: float) -> float:
    """
    Вычисляет оценку интонации по отклонению в центах.
    Малые отклонения почти не штрафуются, большие снижают балл экспоненциально.
    Формула: score = exp(-|cents| / 25)
    """
    return float(1 / (1 + np.exp(abs(deviation_cents) / 40.0)))


def compute_stability(f0_segment: np.ndarray) -> float:
    return 1.0 - (np.std(f0_segment) / np.mean(f0_segment))


class PitchAnalyzerExtended(PitchAnalyzer):
    def analyze_extended(self, segments: List[Dict[str, Any]], f0: np.ndarray = None, times: np.ndarray = None,
                         tempo: float = 120.0, reference: List[str] = None) -> Dict[str, Any]:
        base_report = super().analyze(segments)

        deviations = compute_pitch_deviation(base_report["notes"])
        vibrato = analyze_vibrato(
            f0, times) if f0 is not None and times is not None else {}
        timing = analyze_timing(base_report["notes"], tempo)
        reference_cmp = compare_to_reference(
            base_report["notes"], reference or [])

        # добавляем интонационные оценки для каждой ноты
        intonation_scores = [
            {
                "note": d["note"],
                "midi": d["midi"],
                "deviation_cents": d["deviation_cents"],
                "intonation_score": compute_intonation_score(d["deviation_cents"]),
                "stability_score": compute_stability(np.array([d["freq"]]))
            }
            for d in deviations
        ]

        # общий score (с учётом pitch accuracy, stability, timing)
        pitch_accuracy = np.mean(
            [s["intonation_score"] for s in intonation_scores]) if intonation_scores else 0.0
        stability = base_report["statistics"]["avg_stability"]
        timing_score = timing["avg_duration_ratio"]
        overall = compute_overall_score(
            pitch_accuracy, stability, timing_score)

        base_report.update({
            "notes_with_deviation": deviations,
            "intonation_scores": intonation_scores,
            "vibrato": vibrato,
            "timing_accuracy": timing,
            "reference_comparison": reference_cmp,
            "overall_score": overall
        })
        return base_report

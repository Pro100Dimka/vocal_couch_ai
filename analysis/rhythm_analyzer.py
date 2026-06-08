# analysis/rhythm_analyzer.py
import numpy as np
import math


class RhythmAnalyzer:
    def __init__(self, reference_tempo: float = None, beat_grid: list = None):
        self.reference_tempo = reference_tempo
        self.beat_grid = beat_grid

    # === Utility ===
    def _nearest_beat(self, time: float):
        if not self.beat_grid:
            if not self.reference_tempo:
                return None
            beat_interval = 60.0 / self.reference_tempo
            return round(time / beat_interval) * beat_interval
        return min(self.beat_grid, key=lambda b: abs(b - time))

    def _exp_ratio(self, error_beats: float, k: float = 0.8) -> float:
        return 1 - math.exp(-k * abs(error_beats))

    def _duration_score(self, error_beats: float) -> float:
        return math.exp(-1.2 * abs(error_beats))

    def _timing_score(self, offset_sec: float, beat_interval: float) -> float:
        e = abs(offset_sec) / beat_interval
        return math.exp(-1.5 * e)

    def _severity_label(self, score: float) -> str:
        if score < 0.3:
            return "low"
        elif score < 0.6:
            return "medium"
        else:
            return "high"

    def _dynamic_expected_duration(self, idx: int, beat_interval: float):
        if self.beat_grid and idx < len(self.beat_grid) - 1:
            return self.beat_grid[idx+1] - self.beat_grid[idx]
        return beat_interval

    # === Main analysis ===
    def analyze(self, notes: list):
        if not notes:
            return {}

        attack_offsets, mistakes = [], []
        beat_interval = 60.0 / self.reference_tempo if self.reference_tempo else 0.5

        duration_scores, timing_scores = [], []
        timing_errors_count, duration_errors_count, segmentation_errors_count = 0, 0, 0

        for idx, note in enumerate(notes):
            start, dur = note["start"], note["duration"]
            note_errors = []

            # --- Timing error (onset only) ---
            nearest_beat = self._nearest_beat(start)
            offset = 0.0
            if nearest_beat is not None:
                offset = start - nearest_beat
                attack_offsets.append(offset)

                if offset > 0.25 * beat_interval:
                    note_errors.append(
                        {"type": "timing_error", "value": "late_attack_strong"})
                    timing_errors_count += 1
                elif offset > 0.1 * beat_interval:
                    note_errors.append(
                        {"type": "timing_error", "value": "late_attack_slight"})
                    timing_errors_count += 1
                elif offset < -0.25 * beat_interval:
                    note_errors.append(
                        {"type": "timing_error", "value": "early_attack_strong"})
                    timing_errors_count += 1
                elif offset < -0.1 * beat_interval:
                    note_errors.append(
                        {"type": "timing_error", "value": "early_attack_slight"})
                    timing_errors_count += 1

                timing_scores.append(self._timing_score(offset, beat_interval))

            # --- Duration error (relative to beat grid) ---
            expected_duration = self._dynamic_expected_duration(
                idx, beat_interval)
            duration_error_beats_raw = (
                dur - expected_duration) / beat_interval if beat_interval else 0.0
            duration_error_beats_norm = max(
                min(duration_error_beats_raw, 2.0), -2.0)

            duration_scores.append(
                self._duration_score(duration_error_beats_raw))
            severity_score = self._exp_ratio(duration_error_beats_raw)
            severity = self._severity_label(severity_score)

            # Confidence gating: если ошибка слишком большая → подозрение на segmentation
            if abs(duration_error_beats_raw) > 0.5:
                if abs(duration_error_beats_raw) > 1.5:
                    note_errors.append(
                        {"type": "segmentation_error", "value": "offset_detection_failed"})
                    segmentation_errors_count += 1
                else:
                    note_errors.append({
                        "type": "duration_error",
                        "value": "long_hold" if duration_error_beats_raw > 0 else "short_cut"
                    })
                    duration_errors_count += 1

            if note_errors:
                mistakes.append({
                    "note_index": idx,
                    "errors": note_errors,
                    "offset_sec": round(float(offset), 3),
                    "expected_duration_beats": round(float(expected_duration / beat_interval), 3),
                    "duration_error_beats_raw": round(float(duration_error_beats_raw), 3),
                    "duration_error_beats_norm": round(float(duration_error_beats_norm), 3),
                    "severity_score": round(float(severity_score), 3),
                    "severity": severity
                })

        # --- Метрики ---
        avg_offset = np.mean(attack_offsets) if attack_offsets else 0.0
        avg_abs_offset = np.mean(
            [abs(o) for o in attack_offsets]) if attack_offsets else 0.0

        late_ratio = sum(1 for o in attack_offsets if o > 0.25 *
                         beat_interval) / len(attack_offsets) if attack_offsets else 0.0
        early_ratio = sum(1 for o in attack_offsets if o < -0.25 *
                          beat_interval) / len(attack_offsets) if attack_offsets else 0.0
        on_time_ratio = 1.0 - late_ratio - early_ratio if attack_offsets else 0.0

        timing_accuracy = np.mean(timing_scores) if timing_scores else 0.0
        duration_accuracy = np.median(
            duration_scores) if duration_scores else 0.0
        stability_score = 1.0 - \
            (np.std(attack_offsets) / beat_interval) if attack_offsets else 0.0

        beat_alignment = 1.0 - \
            (avg_abs_offset / beat_interval) if attack_offsets else 0.0

        final_score = 0.4 * timing_accuracy + 0.4 * \
            duration_accuracy + 0.2 * stability_score

        # --- Patterns ---
        patterns = []
        if late_ratio > 0.3:
            patterns.append("consistently_late_attacks")
        if early_ratio > 0.3:
            patterns.append("consistently_early_attacks")
        if duration_errors_count > len(notes) * 0.5:
            patterns.append("unstable_note_length")
        if segmentation_errors_count > 0:
            patterns.append("segmentation_instability")

        return {
            "tempo": self.reference_tempo,
            "timing_accuracy": round(float(timing_accuracy), 2),
            "duration_accuracy": round(float(duration_accuracy), 2),
            "stability_score": round(float(stability_score), 2),
            "beat_alignment": round(float(beat_alignment), 2),
            "final_score": round(float(final_score), 2),
            "attack_timing": {
                "avg_offset_sec": round(float(avg_offset), 3),
                "avg_abs_offset_sec": round(float(avg_abs_offset), 3),
                "late_ratio": round(float(late_ratio), 2),
                "early_ratio": round(float(early_ratio), 2),
                "on_time_ratio": round(float(on_time_ratio), 2),
            },
            "mistakes": mistakes,
            "summary": {
                "timing_errors": timing_errors_count,
                "duration_errors": duration_errors_count,
                "segmentation_errors": segmentation_errors_count
            },
            "patterns": patterns
        }

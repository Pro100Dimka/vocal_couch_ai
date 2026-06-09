# analysis/score_engine.py
import numpy as np


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


class ScoreEngine:
    def analyze(self, audio, pitch_stream, segments, rhythm_stream, voice_quality, embeddings):
        note_segments = segments["note_segments"]

        segment_scores = []
        weighted_sum, total_duration = 0.0, 0.0

        pitch_scores, stability_scores, control_scores, energy_scores, rhythm_scores = [], [], [], [], []

        for seg in note_segments:
            duration = float(seg["duration"])
            total_duration += duration

            # Pitch accuracy
            error_cents = abs(float(seg.get("deviation_cents", 0.0)))
            pitch_score = clamp(100.0 - error_cents / 10.0)

            # Stability
            pitch_std = float(seg.get("pitch_std", 0.0))
            stability_score = clamp(100.0 - pitch_std * 2.0)

            # Control
            attack = abs(float(seg.get("attack", 0.0)))
            decay = abs(float(seg.get("decay", 0.0)))
            jitter = 1.0 - float(seg.get("voiced_ratio", 1.0))
            control_score = clamp(
                100.0 - (attack * 40 + decay * 30 + jitter * 30))

            # Energy
            energy_score = clamp(float(seg.get("energy", 0.0)) * 100.0)

            # Segment score
            seg_score = clamp(
                0.35 * pitch_score +
                0.25 * stability_score +
                0.25 * control_score +
                0.15 * energy_score
            )

            weighted_sum += seg_score * duration

            issues = []
            if pitch_score < 60:
                issues.append("pitch inaccurate")
            if stability_score < 60:
                issues.append("unstable pitch")
            if control_score < 60:
                issues.append("weak control")
            if energy_score < 30:
                issues.append("low energy")

            segment_scores.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "score": round(float(seg_score), 2),
                "main_issues": issues
            })

            pitch_scores.append(pitch_score)
            stability_scores.append(stability_score)
            control_scores.append(control_score)
            energy_scores.append(energy_score)

            # Rhythm
            rhythm_score = 70.0
            if rhythm_stream and "intervals" in rhythm_stream:
                avg_interval = float(np.mean(rhythm_stream["intervals"]))
                rhythm_score = clamp(
                    100.0 - abs(duration - avg_interval) * 10.0)
            rhythm_scores.append(rhythm_score)

        # Components (все значения приведены к float)
        components = {
            "pitch_accuracy": float(np.mean(pitch_scores)) if pitch_scores else 0.0,
            "stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
            "control": float(np.mean(control_scores)) if control_scores else 0.0,
            "intonation": float(np.mean(pitch_scores)) if pitch_scores else 0.0,
            "rhythm": float(np.mean(rhythm_scores)) if rhythm_scores else 0.0
        }

        # Total score = duration‑weighted среднее сегментов
        total_score = weighted_sum / total_duration if total_duration > 0 else 0.0

        # Feedback (мягкие пороги)
        feedback = []
        if sum(1 for s in segment_scores if "pitch inaccurate" in s["main_issues"]) >= len(segment_scores) * 0.10:
            feedback.append("Слишком неточная интонация")
        if sum(1 for s in segment_scores if "unstable pitch" in s["main_issues"]) >= len(segment_scores) * 0.10:
            feedback.append("Слишком нестабильные ноты")
        if sum(1 for s in segment_scores if "weak control" in s["main_issues"]) >= len(segment_scores) * 0.10:
            feedback.append("Проблемы с контролем дыхания")
        if sum(1 for s in segment_scores if "low energy" in s["main_issues"]) >= len(segment_scores) * 0.15:
            feedback.append("Недостаточная энергия исполнения")

        return {
            "total_score": round(float(total_score), 2),
            "components": components,
            "segment_scores": segment_scores,
            "feedback": feedback
        }

import numpy as np
from collections import Counter


class TensionAnalyzer:
    def __init__(self, window_size=2.0):
        self.window_size = window_size

        self.weights = {
            "pitch_instability": 0.30,
            "attack_pressure": 0.20,
            "sustain_instability": 0.20,
            "breath_irregularity": 0.15,
            "transition_instability": 0.15,
        }

    # ================= MAIN =================

    def analyze(self, input_data):
        waveform = np.array(input_data["waveform"])
        sr = input_data["sr"]
        f0 = np.array(input_data["f0"])
        segments = input_data.get("segments", [])

        f0_clean = f0[f0 > 0]

        sub_scores = {
            "pitch_instability": self._pitch_instability(f0_clean),
            "attack_pressure": self._attack_pressure(waveform, sr),
            "sustain_instability": self._sustain_instability(f0_clean),
            "breath_irregularity": self._breath_irregularity(waveform),
            "transition_instability": self._transition_instability(f0_clean),
        }

        tension_score = self._compute_tension(sub_scores)

        flags, events = self._detect_flags(sub_scores, segments, waveform, sr)
        tension_segments = self._build_segments(waveform, sr, events)

        interpretation = self._interpret(sub_scores, tension_score, flags)

        return {
            "tension_score": round(tension_score, 2),
            "sub_scores": sub_scores,
            "tension_flags": flags,
            "tension_events": events,
            "tension_segments": tension_segments,
            "interpretation": interpretation,
        }

    # ================= CORE =================

    def _compute_tension(self, sub_scores):
        tension_parts = {k: (100 - v) for k, v in sub_scores.items()}

        score = sum(
            tension_parts[k] * self.weights[k]
            for k in self.weights
        )

        return float(np.clip(score, 0, 100))

    # ================= METRICS =================

    def _pitch_instability(self, f0):
        if len(f0) < 10:
            return 50.0

        log_f0 = np.log(f0 + 1e-6)

        jitter = np.mean(np.abs(np.diff(log_f0)))
        drift = np.std(log_f0)

        score = 100 - (jitter * 300 + drift * 120)
        return float(np.clip(score, 0, 100))

    def _attack_pressure(self, waveform, sr):
        envelope = np.abs(waveform)
        frame = int(sr * 0.05)

        diffs = []

        for i in range(0, len(envelope) - frame, frame):
            chunk = envelope[i:i + frame]
            if len(chunk) < 10:
                continue

            attack = np.mean(chunk[:len(chunk)//3])
            sustain = np.mean(chunk[-len(chunk)//3:])

            diffs.append(abs(attack - sustain))

        if not diffs:
            return 60.0

        score = 100 - np.mean(diffs) * 120
        return float(np.clip(score, 0, 100))

    def _sustain_instability(self, f0):
        if len(f0) < 10:
            return 50.0

        diff = np.diff(f0)
        instability = np.std(diff)

        score = 100 - instability * 200
        return float(np.clip(score, 0, 100))

    def _breath_irregularity(self, waveform):
        frame = 1024

        rms = [
            np.sqrt(np.mean(waveform[i:i+frame]**2))
            for i in range(0, len(waveform) - frame, frame)
        ]

        if len(rms) < 3:
            return 60.0

        rms = np.array(rms)
        rms_norm = rms / (np.mean(rms) + 1e-6)

        instability = np.mean(np.abs(np.diff(rms_norm)))

        score = 100 - instability * 200
        return float(np.clip(score, 0, 100))

    def _transition_instability(self, f0):
        if len(f0) < 20:
            return 60.0

        log_f0 = np.log(f0 + 1e-6)
        diff = np.diff(log_f0)

        score = 100 - np.mean(np.abs(diff)) * 250
        return float(np.clip(score, 0, 100))

    # ================= EVENTS =================

    def _detect_flags(self, scores, segments, waveform, sr):
        flags = []
        events = []

        if scores["sustain_instability"] < 60:
            flags.append("unstable_sustain")

        if scores["transition_instability"] < 60:
            flags.append("rough_transitions")

        if scores["pitch_instability"] < 60:
            flags.append("pitch_instability")

        if scores["attack_pressure"] < 60:
            flags.append("high_attack_pressure")

        # attack events (real signal-based, not fake loop)
        envelope = np.abs(waveform)
        frame = int(sr * 0.05)

        for i in range(0, len(envelope) - frame, frame):
            chunk = envelope[i:i + frame]
            if len(chunk) < 10:
                continue

            attack = np.mean(chunk[:len(chunk)//3])
            sustain = np.mean(chunk[-len(chunk)//3:])

            if abs(attack - sustain) > np.mean(envelope) * 1.5:
                events.append({
                    "type": "attack_peak",
                    "start": i / sr,
                    "end": (i / sr) + 0.03,
                    "severity": float(np.clip(abs(attack - sustain), 0, 1))
                })

        return flags, events

    # ================= SEGMENTS =================

    def _build_segments(self, waveform, sr, events):
        duration = len(waveform) / sr

        segments = []
        start = 0

        while start < duration:
            end = min(duration, start + self.window_size)

            window_events = [
                e for e in events
                if start <= e["start"] <= end
            ]

            if window_events:
                dominant = Counter(
                    [e["type"] for e in window_events]
                ).most_common(1)[0][0]

                tension = min(1.0, len(window_events) * 0.25)
            else:
                dominant = "stable"
                tension = 0.0

            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "tension": round(tension * 100, 2),
                "dominant_issue": dominant
            })

            start += self.window_size

        return segments

    # ================= INTERPRET =================

    def _interpret(self, scores, tension, flags):
        if tension < 30:
            base = "low vocal tension"
        elif tension < 60:
            base = "moderate vocal tension"
        else:
            base = "high vocal tension"

        reasons = []

        mapping = {
            "unstable_sustain": "нестабильное удержание",
            "rough_transitions": "переходы между нотами",
            "pitch_instability": "нестабильный pitch",
            "high_attack_pressure": "резкие атаки"
        }

        for f in flags:
            if f in mapping:
                reasons.append(mapping[f])

        return base + (". Причины: " + "; ".join(reasons) if reasons else "")

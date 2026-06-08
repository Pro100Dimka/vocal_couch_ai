import numpy as np
from collections import deque


class StyleAnalyzer:
    def __init__(self, frame_size=1024, hop_size=512, smooth_window=3):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.smooth_window = smooth_window

        self.styles = ["breathy", "powerful", "dynamic", "clear", "dense"]

    # ================= MAIN =================
    def analyze(self, input_data):
        waveform = np.array(input_data["waveform"])
        sr = input_data["sr"]

        frames = self._frame_signal(waveform)

        frame_features = [self._extract_features(f) for f in frames]

        style_probs = [self._frame_to_style(f) for f in frame_features]

        smoothed = self._smooth_styles(style_probs)

        distribution = self._aggregate_distribution(smoothed)

        primary_style, confidence = self._get_primary_style(distribution)

        stability = self._compute_stability(smoothed)

        segments = self._build_segments(smoothed, sr)

        return {
            "sub_scores": self._mean_features(frame_features),
            "style_profile": {
                "primary_style": primary_style,
                "confidence": round(confidence, 2),
                "distribution": distribution,
                "stability": round(stability, 2),
            },
            "style_segments": segments,
        }

    # ================= SIGNAL PROCESSING =================
    def _frame_signal(self, waveform):
        frames = []
        for i in range(0, len(waveform) - self.frame_size, self.hop_size):
            frames.append(waveform[i:i + self.frame_size])
        return frames

    # ================= FEATURES =================
    def _extract_features(self, frame):
        eps = 1e-8

        rms = np.sqrt(np.mean(frame ** 2))
        energy = rms

        zero_crossings = np.mean(np.abs(np.diff(np.sign(frame))))

        dynamics = np.std(frame)

        clarity = 1.0 / (zero_crossings + eps)

        breathiness = np.mean(np.abs(frame[: len(frame)//4])) / (rms + eps)

        density = np.count_nonzero(np.abs(frame) > 0.02) / len(frame)

        return {
            "vocal_density": float(density),
            "energy_level": float(energy),
            "dynamics_range": float(dynamics),
            "articulation_clarity": float(clarity),
            "breathiness": float(breathiness),
        }

    # ================= STYLE MODEL =================
    def _frame_to_style(self, f):
        raw = np.array([
            f["breathiness"],        # breathy
            f["energy_level"],       # powerful
            f["dynamics_range"],     # dynamic
            f["articulation_clarity"],  # clear
            f["vocal_density"],      # dense
        ])

        # log scaling (важно для стабильности)
        raw = np.log1p(raw * 10)

        # softmax (probabilities)
        exp = np.exp(raw - np.max(raw))
        probs = exp / np.sum(exp)

        return probs

    # ================= SMOOTHING =================
    def _smooth_styles(self, probs_list):
        smoothed = []
        window = deque(maxlen=self.smooth_window)

        for p in probs_list:
            window.append(p)
            avg = np.mean(window, axis=0)
            smoothed.append(avg)

        return smoothed

    # ================= DISTRIBUTION =================
    def _aggregate_distribution(self, smoothed):
        avg = np.mean(smoothed, axis=0)

        total = np.sum(avg) + 1e-8
        dist = avg / total

        return {
            style: float(dist[i])
            for i, style in enumerate(self.styles)
        }

    # ================= PRIMARY STYLE =================
    def _get_primary_style(self, dist):
        primary = max(dist, key=dist.get)
        confidence = dist[primary] * 100
        return primary, confidence

    # ================= STABILITY =================
    def _compute_stability(self, smoothed):
        arr = np.array(smoothed)
        var = np.mean(np.var(arr, axis=0))
        stability = 100 - var * 200
        return float(np.clip(stability, 0, 100))

    # ================= SEGMENTS =================
    def _build_segments(self, smoothed, sr):
        segments = []
        step_time = self.hop_size / sr

        prev_style = None
        start = 0

        for i, p in enumerate(smoothed):
            style_idx = int(np.argmax(p))
            style = self.styles[style_idx]

            if style != prev_style:
                if prev_style is not None:
                    segments.append({
                        "start": round(start, 2),
                        "end": round(i * step_time, 2),
                        "style": prev_style
                    })
                start = i * step_time
                prev_style = style

        if prev_style is not None:
            segments.append({
                "start": round(start, 2),
                "end": round(len(smoothed) * step_time, 2),
                "style": prev_style
            })

        return segments

    # ================= FEATURES SUMMARY =================
    def _mean_features(self, features):
        keys = features[0].keys()
        return {
            k: float(np.mean([f[k] for f in features]))
            for k in keys
        }



class ScoreEngine:
    def __init__(self):
        pass

    def analyze(self, pitch, style, tension):
        return self.aggregate(pitch, style, tension)

    def analyze_reference(self, user, reference):

        pitch_score = self._compare_pitch(user["pitch"], reference["pitch"])
        style_score = self._compare_style(user["style"], reference["style"])
        tension_score = self._compare_tension(
            user["tension"], reference["tension"])

        final_score = (
            pitch_score * 0.4 +
            style_score * 0.3 +
            tension_score * 0.3
        )

        return {
            "final_score": round(final_score, 2),
            "modules": {
                "pitch": pitch_score,
                "style": style_score,
                "tension": tension_score
            }
        }

    # -------------------------
    # INTERNAL SCORING LOGIC
    # -------------------------

    def _compare_pitch(self, user, ref):
        user_score = user.get("pitch_quality", {}).get("score", 50)
        ref_score = ref.get("pitch_quality", {}).get("score", 50)

        return max(0, 100 - abs(user_score - ref_score))

    def _compare_style(self, user, ref):
        user_style = user.get("style_profile", {}).get("confidence", 50)
        ref_style = ref.get("style_profile", {}).get("confidence", 50)

        return max(0, 100 - abs(user_style - ref_style))

    def _compare_tension(self, user, ref):
        user_t = user.get("tension_score", 50)
        ref_t = ref.get("tension_score", 50)

        return max(0, 100 - abs(user_t - ref_t))

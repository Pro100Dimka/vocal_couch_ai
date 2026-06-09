# coach/segment_analyzer.py
def analyze_segments(segment_scores, threshold=85.0):
    """Ищет слабые сегменты и возвращает таймкоды"""
    weak_segments = []
    for seg in segment_scores:
        if seg["score"] < threshold or seg["main_issues"]:
            weak_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "score": seg["score"],
                "issues": seg["main_issues"]
            })
    return weak_segments

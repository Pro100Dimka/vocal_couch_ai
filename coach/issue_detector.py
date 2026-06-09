# coach/issue_detector.py
from collections import Counter


def detect_issues(segment_scores):
    # собираем все проблемы из сегментов
    all_issues = []
    for seg in segment_scores:
        all_issues.extend(seg.get("main_issues", []))

    counter = Counter(all_issues)
    # сортируем по частоте
    sorted_issues = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    return {
        "issue_counts": dict(counter),
        "top_issues": [issue for issue, _ in sorted_issues]
    }

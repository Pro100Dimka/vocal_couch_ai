# coach/summary_builder.py

def build_summary(analysis_report, issues, recommendations, exercises, segment_feedback):
    # 1. Общий балл
    overall_score = round(float(analysis_report["total_score"]), 2)

    # 2. Сильные стороны (компоненты > 90)
    strengths = [
        k for k, v in analysis_report["components"].items() if v >= 90
    ]

    # 3. Главные проблемы (топ-3)
    weaknesses = issues["top_issues"][:3]

    # 4. Суммаризация сегментов
    # Берём топ-5 самых слабых сегментов
    weak_segments_sorted = sorted(segment_feedback, key=lambda s: s["score"])
    top_weak_segments = weak_segments_sorted[:5]

    # Считаем количество проблем по типам
    issue_counts = issues["issue_counts"]

    # 5. Формируем компактный план тренировки
    plan = []
    for ex in exercises:
        plan.append(f"{ex['name']} — {ex['duration']}")

    summary = {
        "overall_score": overall_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "issue_counts": issue_counts,
        "recommendations": recommendations,
        "exercises": plan,
        "weak_segments": top_weak_segments
    }

    return summary

# coach/feedback_generator.py
from coach.issue_detector import detect_issues
from coach.segment_analyzer import analyze_segments
from coach.rules import get_recommendations
from coach.exercise_builder import build_exercises
from coach.summary_builder import build_summary
from coach.types import FeedbackReport


class FeedbackGenerator:
    def generate(self, analysis_report):
        # 1. Определяем глобальные проблемы
        issues = detect_issues(analysis_report["segment_scores"])

        # 2. Анализируем сегменты (таймкоды, опасные зоны)
        segment_feedback = analyze_segments(analysis_report["segment_scores"])

        # 3. Получаем рекомендации по каждой проблеме
        recommendations = []
        for issue in issues["top_issues"]:
            rec = get_recommendations(issue)
            recommendations.append(rec)

        # 4. Генерируем упражнения
        exercises = []
        for issue in issues["top_issues"]:
            ex = build_exercises(issue)
            exercises.extend(ex)

        # 5. Собираем финальный отчёт
        summary = build_summary(
            analysis_report=analysis_report,
            issues=issues,
            recommendations=recommendations,
            exercises=exercises,
            segment_feedback=segment_feedback
        )

        return FeedbackReport(
            total_score=analysis_report["total_score"],
            components=analysis_report["components"],
            issues=issues,
            recommendations=recommendations,
            exercises=exercises,
            segment_feedback=segment_feedback,
            summary=summary
        )

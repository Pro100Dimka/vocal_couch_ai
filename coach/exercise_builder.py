# coach/exercise_builder.py
def build_exercises(issue):
    exercises = []
    if issue == "low energy":
        exercises.append({"name": "Дыхание на счёт", "duration": "5 мин"})
        exercises.append({"name": "Громкие гласные", "duration": "5 мин"})
    elif issue == "weak control":
        exercises.append(
            {"name": "Удержание нот 10 секунд", "duration": "5 мин"})
    elif issue == "unstable pitch":
        exercises.append(
            {"name": "Пение гамм с тюнером", "duration": "10 мин"})
    elif issue == "rhythm":
        exercises.append({"name": "Пение под метроном", "duration": "10 мин"})
    else:
        exercises.append(
            {"name": "Общая вокальная разминка", "duration": "5 мин"})
    return exercises

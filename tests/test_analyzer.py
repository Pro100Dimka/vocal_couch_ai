# tests/test_analyzer.py
from audio.loader import load_audio
from features.pitch import get_pitch_features
from analysis.pitch_analyzer import (
    analyze_pitch,
)

# Загружаем тестовый аудиофайл
y, sr = load_audio("audio_samples/my_voice.wav")

# Извлекаем признаки
pitch = get_pitch_features(y, sr, method="yin")

# Анализируем
result = analyze_pitch(pitch)

# Проверяем структуру результата
assert "dominant_note_midi" in result
assert "dominant_note_name" in result
assert "confidence" in result
assert "stability" in result
assert "score" in result
assert "segments" in result

# Дополнительные проверки
assert 0.0 <= result["confidence"] <= 1.0
assert 0.0 <= result["score"] <= 100.0

# Вывод для наглядности
print("Dominant note (MIDI):", result["dominant_note_midi"])
print("Dominant note:", result["dominant_note_name"])
print("Confidence:", result["confidence"])
print("Stability:", result["stability"])
print("Score:", result["score"])

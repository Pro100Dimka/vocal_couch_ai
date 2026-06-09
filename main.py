from audio.loader import AudioLoader
from audio.preprocessor import AudioPreprocessor
from features.pitch import get_pitch
from utils.file_ops import remove_old
from utils.logger import log
from features.rhythm import get_rhythm
from features.voice_quality import get_voice_quality
from features.embeddings import get_embeddings
from features.segmentation import get_segmentation_features
from analysis.score_engine import ScoreEngine

# Coach modules
from coach.feedback_generator import FeedbackGenerator


def main():
    remove_old([])
    loader = AudioLoader()
    preprocessor = AudioPreprocessor()

    # 1. Загружаем и обрабатываем аудио
    audio = loader.load("audio_samples/my_voice.wav")
    processed = preprocessor.process(audio)

    # 2. Извлекаем фичи
    embeddings = get_embeddings(processed)
    voice_quality_stream = get_voice_quality(processed)
    rhythm_stream = get_rhythm(processed, resolution=0.5)
    pitch = get_pitch(processed, method="yin")
    segments = get_segmentation_features(pitch)

    # 3. Анализируем результат
    score_engine = ScoreEngine()
    analysis_report = score_engine.analyze(
        audio=processed,
        pitch_stream=pitch,
        segments=segments,
        rhythm_stream=rhythm_stream,
        voice_quality=voice_quality_stream,
        embeddings=embeddings
    )

    # 4. Генерируем коуч‑отчёт
    feedback_generator = FeedbackGenerator()
    coach_report = feedback_generator.generate(analysis_report)
    log("\n=== FINAL COACH REPORT ===")
    log(coach_report.summary)


if __name__ == "__main__":
    main()

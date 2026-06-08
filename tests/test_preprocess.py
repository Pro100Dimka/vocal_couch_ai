# test_preprocess.py
import soundfile as sf
from audio.loader import load_audio, AudioData
from audio.preprocessing import preprocess_audio


def main():
    audio: AudioData = load_audio("audio_samples/my_voice.wav")
    print(f"[LOADER] Длительность: {audio.duration:.2f} сек")
    print(f"[LOADER] Амплитуда: min={audio.min_amplitude:.3f}, "
          f"max={audio.max_amplitude:.3f}, mean={audio.mean_amplitude:.3f}")
    processed_waveform = preprocess_audio(audio.waveform, audio.sample_rate)
    processed = AudioData(
        waveform=processed_waveform,
        sample_rate=audio.sample_rate,
        duration=len(processed_waveform) / audio.sample_rate,
        samples=len(processed_waveform),
        min_amplitude=float(processed_waveform.min()),
        max_amplitude=float(processed_waveform.max()),
        mean_amplitude=float(processed_waveform.mean()),
        std_amplitude=float(processed_waveform.std()),
    )
    print(f"[PREPROCESS] Длительность: {processed.duration:.2f} сек")
    print(f"[PREPROCESS] Амплитуда: min={processed.min_amplitude:.3f}, "
          f"max={processed.max_amplitude:.3f}, mean={processed.mean_amplitude:.3f}")

    # === Этап 3. Сохранение результата ===
    sf.write("audio_samples/my_voice_processed.wav",
             processed.waveform, processed.sample_rate)
    print("[INFO] Файл сохранён: audio_samples/my_voice_processed.wav")


if __name__ == "__main__":
    main()

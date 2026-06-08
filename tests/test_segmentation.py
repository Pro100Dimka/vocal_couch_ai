# tests/test_segmentation.py
import numpy as np
from audio.loader import load_audio, AudioData
from audio.preprocessing import preprocess_audio
from features.segmentation import get_segmentation_features
from features.pitch import get_pitch_stream


def main():
    audio: AudioData = load_audio("audio_samples/my_voice.wav")
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

    pitch_stream = get_pitch_stream(processed, method="yin")

    segments = get_segmentation_features(
        f0=np.array(pitch_stream["f0"]),
        voiced=np.array(pitch_stream["voiced"]),
        times=np.array(pitch_stream["times"])
    )

    print("Detected segments:")
    for seg in segments:
        print(seg)


if __name__ == "__main__":
    main()

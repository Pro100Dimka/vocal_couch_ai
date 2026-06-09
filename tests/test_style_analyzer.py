# tests/test_style_analyzer.py.py
import numpy as np
import json
from utils.logger import log
from analysis.style_analyzer import StyleAnalyzer
from audio.loader import load_audio, AudioData
from audio.preprocessor import preprocess_audio
from features.segmentation import get_segmentation_features
from features.pitch import get_pitch_stream
from utils.file_ops import remove_old
from analysis.pitch_analyzer import (
    hz_to_midi,
    midi_to_note_name,
    merge_short_notes,
    smooth_notes
)


def main():
    remove_old([])
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

    raw_segments = get_segmentation_features(
        f0=np.array(pitch_stream["f0"]),
        voiced=np.array(pitch_stream["voiced"]),
        times=np.array(pitch_stream["times"])
    )

    segments = []
    for seg in raw_segments:
        duration = seg["end"] - seg["start"]
        midi = hz_to_midi(seg["f0_mean"])
        segments.append({
            "note": midi_to_note_name(midi),
            "freq": seg["f0_mean"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": duration,
            "stability": seg["f0_stability"],
            "type": seg.get("type", "sung_note")
        })

    segments = merge_short_notes(segments, min_duration=0.08)
    segments = smooth_notes(segments)


# ...
    tension_analyzer = StyleAnalyzer()
    tension_report = tension_analyzer.analyze({
        "waveform": processed.waveform,
        "sr": processed.sample_rate,
        "f0": np.array(pitch_stream["f0"]),
        "times": np.array(pitch_stream["times"]),
        "segments": segments
    })
    print("\n=== TENSION ANALYSIS REPORT ===")
    print(json.dumps(tension_report, indent=2))
    log("\n=== TENSION ANALYSIS REPORT ===")
    log(json.dumps(tension_report, indent=2))


if __name__ == "__main__":
    main()

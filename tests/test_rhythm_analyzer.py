# tests/rhythm_analyzer.py
import numpy as np
from audio.loader import load_audio, AudioData
from audio.preprocessor import preprocess_audio
from features.segmentation import get_segmentation_features
from features.pitch import get_pitch_stream
from utils.logger import log
from utils.file_ops import remove_old
from analysis.pitch_analyzer import (
    PitchAnalyzerExtended,
    hz_to_midi,
    midi_to_note_name,
    merge_short_notes,
    smooth_notes
)
from analysis.rhythm_analyzer import RhythmAnalyzer
import json


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

    # 🔑 Преобразуем в NoteSegment формат
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

    ext_analyzer = PitchAnalyzerExtended()
    extended_report = ext_analyzer.analyze_extended(
        segments=segments,
        f0=np.array(pitch_stream["f0"]),
        times=np.array(pitch_stream["times"]),
        tempo=120.0,
        reference=["C4", "E4", "G4"]
    )
    rhythm_analyzer = RhythmAnalyzer(reference_tempo=120.0)

    # Берём список нот из extended_report["notes"]
    rhythm_report = rhythm_analyzer.analyze([
        {"start": n["start"], "end": n["end"], "duration": n["duration"]}
        for n in extended_report["notes"]   # <-- исправлено
        if n["type"] == "sung_note"
    ])

    log(json.dumps(rhythm_report, indent=2))


if __name__ == "__main__":
    main()

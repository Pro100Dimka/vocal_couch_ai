import numpy as np
import json

from utils.logger import log

from audio.loader import load_audio, AudioData
from audio.preprocessing import preprocess_audio

from features.segmentation import get_segmentation_features
from features.pitch import get_pitch_stream

from analysis.pitch_analyzer import (
    PitchAnalyzerExtended,
    hz_to_midi,
    midi_to_note_name,
    merge_short_notes,
    smooth_notes
)

from analysis.style_analyzer import StyleAnalyzer
from analysis.tension_analyzer import TensionAnalyzer
from analysis.score_engine import ScoreEngine


# -------------------------
# BUILD FULL ANALYSIS PIPELINE
# -------------------------

def build_analysis(audio_path: str):

    audio: AudioData = load_audio(audio_path)

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
        midi = hz_to_midi(seg["f0_mean"])

        segments.append({
            "note": midi_to_note_name(midi),
            "freq": seg["f0_mean"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["end"] - seg["start"],
            "stability": seg["f0_stability"],
            "type": seg.get("type", "sung_note")
        })

    segments = merge_short_notes(segments, min_duration=0.08)
    segments = smooth_notes(segments)

    # -------------------------
    # PITCH
    # -------------------------
    pitch_analyzer = PitchAnalyzerExtended()
    pitch_report = pitch_analyzer.analyze_extended(
        segments=segments,
        f0=np.array(pitch_stream["f0"]),
        times=np.array(pitch_stream["times"]),
        tempo=120.0,
        reference=["C4", "E4", "G4"]
    )

    # -------------------------
    # STYLE
    # -------------------------
    style_analyzer = StyleAnalyzer()
    style_report = style_analyzer.analyze({
        "waveform": processed.waveform,
        "sr": processed.sample_rate,
        "f0": np.array(pitch_stream["f0"]),
        "times": np.array(pitch_stream["times"]),
        "segments": segments
    })

    # -------------------------
    # TENSION
    # -------------------------
    tension_analyzer = TensionAnalyzer()
    tension_report = tension_analyzer.analyze({
        "waveform": processed.waveform,
        "sr": processed.sample_rate,
        "f0": np.array(pitch_stream["f0"]),
        "times": np.array(pitch_stream["times"]),
        "segments": segments
    })

    return {
        "pitch": pitch_report,
        "style": style_report,
        "tension": tension_report
    }


# -------------------------
# MAIN TEST
# -------------------------

def main():

    engine = ScoreEngine()

    # 🎤 USER (твоя запись)
    user = build_analysis("audio_samples/my_voice.wav")

    # 🎧 REFERENCE (эталон / песня)
    reference = build_analysis("audio_samples/reference.wav")

    result = engine.analyze_reference(user, reference)

    log("\n=== SCORE ENGINE REPORT ===")
    log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

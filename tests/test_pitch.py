# tests/test_pitch.py
from audio.loader import load_audio
from features.pitch import get_pitch_features

y, sr = load_audio("audio_samples/my_voice.wav")

pitch = get_pitch_features(y, sr, method="yin")

print("Times:", pitch["times"][:10])
print("F0 raw:", pitch["f0_raw"][:10])
print("F0 smooth:", pitch["f0"][:10])
print("MIDI:", pitch["midi"][:10])
print("Voiced:", pitch["voiced"][:10])

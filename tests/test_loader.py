from audio.loader import load_audio
import matplotlib.pyplot as plt

audio_path = "audio_samples/my_voice.wav"  # положи любой вокал сюда

y, sr = load_audio(audio_path)

print("Shape:", y.shape)
print("Sample rate:", sr)
print("Min/Max:", y.min(), y.max())

plt.plot(y[:2000])
plt.title("Audio waveform (first 2000 samples)")
plt.show()

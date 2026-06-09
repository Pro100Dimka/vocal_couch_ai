# audio/preprocessor.py
import numpy as np
import librosa
import scipy.signal
from dataclasses import dataclass
from audio.loader import AudioData


@dataclass
class ProcessedAudioData(AudioData):
    processed: bool = True
    silence_mask: np.ndarray | None = None  # важно для анализа


class AudioPreprocessor:
    def __init__(
        self,
        target_peak: float = 0.95,
        cutoff: float = 80.0,
        top_db: float = 30,
        silence_fill: float = 0.0,   # можно 0.0 или 0.02 (мягкий режим)
        use_hard_silence: bool = False
    ):
        self.target_peak = target_peak
        self.cutoff = cutoff
        self.top_db = top_db
        self.silence_fill = silence_fill
        self.use_hard_silence = use_hard_silence

    # ----------------------------
    # 1. NORMALIZATION
    # ----------------------------
    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (self.target_peak / peak)
        return y

    # ----------------------------
    # 2. DC OFFSET REMOVAL
    # ----------------------------
    def remove_dc_offset(self, y: np.ndarray) -> np.ndarray:
        return y - np.mean(y)

    # ----------------------------
    # 3. SILENCE DETECTION (CRITICAL FIX)
    # ----------------------------
    def remove_silence(self, y: np.ndarray, sr: int):
        intervals = librosa.effects.split(y, top_db=self.top_db)

        mask = np.zeros(len(y), dtype=np.float32)

        for start, end in intervals:
            mask[start:end] = 1.0

        self.last_silence_mask = mask

        if self.use_hard_silence:
            # ❗ полностью убираем тишину (для dataset/training)
            return y * mask
        else:
            # ⚠️ мягкий режим (для analysis / pitch / style)
            return y * mask + (1.0 - mask) * self.silence_fill

    # ----------------------------
    # 4. FILTERING
    # ----------------------------
    def high_pass_filter(self, y: np.ndarray, sr: int) -> np.ndarray:
        sos = scipy.signal.butter(
            N=4,
            Wn=self.cutoff,
            btype="highpass",
            fs=sr,
            output="sos"
        )
        return scipy.signal.sosfilt(sos, y)

    # ----------------------------
    # 5. FINAL NORMALIZATION
    # ----------------------------
    def final_normalize(self, y: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(y))
        if peak > 1.0:
            y = y / peak
        return np.clip(y, -1.0, 1.0)

    # ----------------------------
    # MAIN PIPELINE
    # ----------------------------
    def process(self, audio: AudioData) -> ProcessedAudioData:
        y = audio.waveform.copy()

        y = self.normalize_audio(y)
        y = self.remove_dc_offset(y)
        y = self.remove_silence(y, audio.sample_rate)
        y = self.high_pass_filter(y, audio.sample_rate)
        y = self.final_normalize(y)

        return ProcessedAudioData(
            waveform=y,
            sample_rate=audio.sample_rate,
            duration=len(y) / audio.sample_rate,
            samples=len(y),
            min_amplitude=float(np.min(y)),
            max_amplitude=float(np.max(y)),
            mean_amplitude=float(np.mean(y)),
            std_amplitude=float(np.std(y)),
            processed=True,
            silence_mask=getattr(self, "last_silence_mask", None),
        )

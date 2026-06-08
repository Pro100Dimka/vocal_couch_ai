# features/segmentation.py
import numpy as np
from typing import List, Dict, Optional


def get_segmentation_features(
    f0: np.ndarray,
    voiced: np.ndarray,
    times: np.ndarray,
    energy: Optional[np.ndarray] = None,
    min_duration: float = 0.05,   # минимальная длительность сегмента (50 ms)
    # порог нестабильности f0 (в относительных единицах)
    stability_threshold: float = 0.15
) -> List[Dict]:
    """
    Превращает поток f0 + voiced (+ energy) в список вокальных сегментов.

    Parameters
    ----------
    f0 : np.ndarray
        Массив частот основного тона (Hz).
    voiced : np.ndarray
        Бинарная маска (1 = голос, 0 = тишина/шум).
    times : np.ndarray
        Временная шкала для каждого значения f0.
    energy : np.ndarray, optional
        Амплитуда/громкость по времени.
    min_duration : float
        Минимальная длительность сегмента (сек).
    stability_threshold : float
        Порог относительной нестабильности f0 (MAD/mean).

    Returns
    -------
    segments : List[Dict]
        Список словарей с описанием вокальных событий.
    """

    segments = []
    n = len(f0)

    i = 0
    while i < n:
        if voiced[i] == 1:
            # начало потенциального сегмента
            start_idx = i
            while i < n and voiced[i] == 1:
                i += 1
            end_idx = i - 1

            # извлекаем данные сегмента
            seg_f0 = f0[start_idx:end_idx + 1]
            seg_times = times[start_idx:end_idx + 1]
            seg_energy = energy[start_idx:end_idx +
                                1] if energy is not None else None

            duration = seg_times[-1] - seg_times[0]

            # фильтр по минимальной длительности
            if duration < min_duration:
                continue

            # оценка стабильности f0
            mean_f0 = float(np.mean(seg_f0[seg_f0 > 0])) if np.any(
                seg_f0 > 0) else 0.0
            mad_f0 = float(np.median(np.abs(seg_f0 - mean_f0))
                           ) if mean_f0 > 0 else 0.0
            stability = 1.0 - min(1.0, mad_f0 / (mean_f0 + 1e-6))

            # определение типа сегмента
            if mean_f0 == 0:
                seg_type = "breath/noise"
            elif stability > (1 - stability_threshold):
                seg_type = "sung_note"
            else:
                # проверка на резкие скачки
                jumps = np.sum(np.abs(np.diff(seg_f0)) > mean_f0 * 0.2)
                if jumps > 2:
                    seg_type = "pitch_transition"
                else:
                    seg_type = "break"

            segments.append({
                "start": float(seg_times[0]),
                "end": float(seg_times[-1]),
                "f0_mean": round(mean_f0, 2),
                "f0_stability": round(stability, 3),
                "type": seg_type
            })
        else:
            i += 1

    return segments

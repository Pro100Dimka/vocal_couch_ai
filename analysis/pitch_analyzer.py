import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_number):
    if midi_number is None:
        return None
    midi_number = int(round(midi_number))
    return NOTE_NAMES[midi_number % 12] + str((midi_number // 12) - 1)


def clean_pitch(midi, voiced):
    midi = np.asarray(midi, dtype=np.float32).copy()
    voiced = np.asarray(voiced, dtype=bool)
    midi[~voiced] = np.nan
    return midi


def smooth(x, win=5):
    if len(x) < win:
        return x
    return np.convolve(x, np.ones(win)/win, mode="same")


def segment_notes(pitch, threshold=1.5, min_len=8):
    segments = []
    cur = []

    for i in range(1, len(pitch)):
        if np.isnan(pitch[i]):
            continue

        if len(cur) == 0:
            cur.append(pitch[i])
            continue

        if abs(pitch[i] - pitch[i - 1]) > threshold:
            if len(cur) >= min_len:
                segments.append(np.array(cur))
            cur = [pitch[i]]
        else:
            cur.append(pitch[i])

    if len(cur) >= min_len:
        segments.append(np.array(cur))

    return segments


def get_note(seg):
    seg = seg[np.isfinite(seg)]
    if len(seg) < 5:
        return None

    seg = smooth(seg, 5)
    return int(round(np.mean(seg)))


def weighted_dominant(segments):
    votes = {}

    for seg in segments:
        note = get_note(seg)
        if note is None:
            continue

        weight = len(seg)
        votes[note] = votes.get(note, 0) + weight

    if not votes:
        return None

    return max(votes.items(), key=lambda x: x[1])[0]


def analyze_pitch(data):
    midi = data.get("midi", [])
    voiced = data.get("voiced", [])

    pitch = clean_pitch(midi, voiced)

    segments = segment_notes(pitch)

    melody = []
    lengths = []

    for seg in segments:
        note = get_note(seg)
        if note is not None:
            melody.append(note)
            lengths.append(len(seg))

    dominant = weighted_dominant(segments)

    # ===== FIXED METRICS =====
    valid = len(melody)
    confidence = valid / max(1, len(segments))

    if len(melody) > 0:
        stability = float(np.std(melody))
    else:
        stability = 0.0

    score = float(np.clip(100 - stability * 5, 0, 100) * confidence)

    return {
        "melody": melody,
        "dominant_note_midi": dominant,
        "dominant_note_name": midi_to_note_name(dominant),
        "confidence": confidence,
        "stability": stability,
        "score": score,
        "segments": len(segments)
    }

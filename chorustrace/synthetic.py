"""Deterministic, procedural singing and backing tracks for pipeline checks."""

import json
from pathlib import Path

import numpy as np

from .audio import write_wav

_SAMPLE_RATE = 16_000
_TRAIN_LYRICS = (
    ("hello moon", "we see the light", "the moon is high", "sing with me"),
    ("we sing high", "sing tonight", "hear the moon", "with me today"),
    ("say hello", "we see the moon", "sing a song", "the light is high"),
    ("we see you today", "hear me sing", "a song tonight", "light the sky"),
    ("see the sky", "we hear you", "hello tonight", "a high note"),
    ("sing with you", "hear the note", "today we sing", "the song is high"),
)
_TEST_LYRICS = (
    (
        ("repeat-1", "we sing tonight"),
        ("repeat-2", "we sing tonight"),
        ("changed", "we sing today"),
    ),
    (
        ("line-1", "see the light"),
        ("line-2", "we hear the song"),
        ("line-3", "sing a high note"),
    ),
    (
        ("line-1", "hello sky"),
        ("line-2", "hear me tonight"),
        ("line-3", "we sing a song"),
    ),
)

_PHONES = {
    "a": "AH", "hear": "HH IH R", "hello": "HH EH L OW", "high": "HH AY",
    "is": "IH Z", "light": "L AY T", "me": "M IY", "moon": "M UW N",
    "note": "N OW T", "say": "S EY", "see": "S IY", "sing": "S IH NG",
    "sky": "S K AY", "song": "S AO NG", "the": "DH AH", "today": "T AH D EY",
    "tonight": "T AH N AY T", "we": "W IY", "with": "W IH TH", "you": "Y UW",
}
_VOWELS = {
    "AH": (620, 1310, 2500), "AO": (570, 880, 2500), "EH": (550, 1800, 2490),
    "IH": (410, 1990, 2550), "IY": (310, 2230, 2970), "UW": (330, 870, 2250),
    "AY": (730, 1240, 2560), "EY": (500, 1840, 2460), "OW": (500, 920, 2520),
}
_VOWEL_END = {"AY": "IY", "EY": "IY", "OW": "UW"}
_RESONANTS = {
    "L": (460, 1150, 2600), "M": (240, 1100, 2150), "N": (270, 1650, 2600),
    "NG": (300, 1900, 2730), "R": (430, 1250, 1700), "W": (320, 750, 2180),
    "Y": (310, 2310, 3100),
}
_NOISE = {
    "HH": (1700, 1400, 0.035), "S": (4800, 1700, 0.044), "Z": (4300, 1600, 0.046),
    "TH": (3000, 1600, 0.027), "DH": (2050, 1300, 0.032), "T": (3900, 1700, 0.065),
    "D": (2700, 1700, 0.055), "K": (2300, 950, 0.062),
}
_MELODIES = ((0, 2, 4, 2, 0), (4, 2, 0, 2, 4), (0, 0, 5, 4, 2), (2, 4, 7, 4, 0))


def _rms(samples):
    return np.sqrt(np.mean(samples * samples))


def _frequency(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _formants(harmonics, pitch, centers, timbre):
    distance = (harmonics[:, None] * pitch - timbre * np.asarray(centers)) / [110, 170, 230]
    resonance = np.exp(-0.5 * distance ** 2) @ np.array([1.0, 0.62, 0.41])
    return (0.12 + resonance) / harmonics ** 0.9


def _voice(phase, pitch, centers, end_centers, timbre):
    harmonics = np.arange(1, min(35, int(7500 / pitch)) + 1)
    waves = np.sin(phase[:, None] * harmonics)
    signal = waves @ _formants(harmonics, pitch, centers, timbre)
    if end_centers is not None:
        ending = waves @ _formants(harmonics, pitch, end_centers, timbre)
        blend = np.clip(np.linspace(-0.3, 1.2, len(phase)), 0.0, 1.0)
        signal = (1 - blend) * signal + blend * ending
    return signal / _rms(signal)


def _shaped_noise(rng, size, center, width):
    frequencies = np.fft.rfftfreq(size, d=1 / _SAMPLE_RATE)
    noise = np.fft.rfft(rng.standard_normal(size))
    band = np.exp(-0.5 * ((frequencies - center) / width) ** 2)
    signal = np.fft.irfft(noise * band, n=size)
    return signal / _rms(signal)


def _sing(text, key, melody, timbre, rng):
    phase = rng.uniform(0, 2 * np.pi)
    personality = rng.uniform(0.88, 1.16)
    vibrato = rng.uniform(4.5, 6.1)
    detune = rng.uniform(-0.19, 0.19)
    position = int(rng.uniform(0.06, 0.13) * _SAMPLE_RATE)
    parts = [np.zeros(position)]
    note = 0
    previous_pitch = _frequency(key + melody[0])
    for word in text.split():
        for phone in _PHONES[word].split():
            if phone in _VOWELS:
                pitch = _frequency(key + melody[note % len(melody)] + detune)
                duration = rng.uniform(0.15, 0.23) * (1.32 if note % 3 == 1 else 1.0)
                note += 1
            else:
                pitch = previous_pitch
                duration = rng.uniform(0.055, 0.091)
            size = round(personality * duration * _SAMPLE_RATE)
            t = np.arange(size) / _SAMPLE_RATE
            instantaneous = pitch + (previous_pitch - pitch) * np.exp(-t / 0.022)
            elapsed = t + position / _SAMPLE_RATE
            instantaneous *= 1 + 0.016 * np.sin(2 * np.pi * vibrato * elapsed)
            voice_phase = phase + 2 * np.pi * np.cumsum(instantaneous) / _SAMPLE_RATE
            previous_pitch = pitch
            phase = voice_phase[-1] % (2 * np.pi)
            position += size

            if phone in _VOWELS:
                end_centers = _VOWELS.get(_VOWEL_END.get(phone))
                signal = 0.12 * _voice(voice_phase, pitch, _VOWELS[phone], end_centers, timbre)
            elif phone in _RESONANTS:
                signal = 0.062 * _voice(voice_phase, pitch, _RESONANTS[phone], None, timbre)
            else:
                center, width, loudness = _NOISE[phone]
                signal = loudness * _shaped_noise(rng, size, center, width)
                if phone in ("T", "D", "K"):
                    signal *= np.exp(-((t / t[-1] - 0.73) / 0.15) ** 2)
                if phone in ("Z", "DH", "D"):
                    signal += 0.034 * np.sin(voice_phase)

            edge = np.minimum(t, t[-1] - t) / 0.008
            parts.append(signal * np.clip(edge, 0, 1))

        gap = int(rng.uniform(0.015, 0.04) * _SAMPLE_RATE)
        parts.append(np.zeros(gap))
        position += gap
    parts.append(np.zeros(int(0.10 * _SAMPLE_RATE)))
    return np.concatenate(parts)


def _backing(size, key, tempo, rng):
    t = np.arange(size) / _SAMPLE_RATE
    beat_length = 60 / tempo
    beat_position = t + rng.uniform(0.0, 0.30)
    beat = (beat_position / beat_length).astype(int)
    in_beat = beat_position % beat_length
    root = key - 24 + np.array([0, 5, 7, 5])[(beat // 2) % 4]
    bass_phase = 2 * np.pi * _frequency(root) * in_beat
    bass = (np.sin(bass_phase) + 0.30 * np.sin(2 * bass_phase)) * np.exp(-5.2 * in_beat)
    chord = np.zeros(size)
    for interval in (12, 16, 19, 24):
        phase = 2 * np.pi * _frequency(root + interval) * in_beat
        chord += np.sin(phase) + 0.35 * np.sin(3 * phase) + 0.20 * np.sin(5 * phase)
    chord *= np.exp(-7.6 * in_beat)
    kick_phase = 2 * np.pi * (46 * in_beat + 1.2 * (1 - np.exp(-46 * in_beat)))
    kick = np.sin(kick_phase) * np.exp(-21 * in_beat)
    in_half_beat = beat_position % (beat_length / 2)
    ticks = np.diff(rng.standard_normal(size + 1)) * np.exp(-62 * in_half_beat)
    fade = np.clip(np.minimum(t, t[-1] - t) / 0.012, 0, 1)
    return fade * (0.55 * bass + 0.21 * chord + 0.34 * kick + 0.13 * ticks)


def make_synthetic_sources(output_dir: str | Path, seed: int = 0) -> Path:
    """Write aligned stems, synthetic mixes, and a song-split source JSONL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sources = output_dir / "sources.jsonl"
    train_songs = [
        tuple((f"line-{i}", text) for i, text in enumerate(song, 1))
        for song in _TRAIN_LYRICS
    ]
    with sources.open("w", encoding="utf-8") as manifest:
        for split, songs in (("train", train_songs), ("test", _TEST_LYRICS)):
            for i, song in enumerate(songs, 1):
                song_id = f"{split}-{i:02d}"
                key = int(rng.integers(53, 62))
                melody = _MELODIES[int(rng.integers(len(_MELODIES)))]
                tempo = rng.uniform(96, 131)
                timbre = rng.uniform(0.91, 1.11)
                native_level = rng.uniform(-3.5, 2.0)
                for clip_id, text in song:
                    vocal = _sing(text, key, melody, timbre, rng)
                    accompaniment = _backing(len(vocal), key, tempo, rng)
                    accompaniment *= _rms(vocal) * 10 ** (native_level / 20) / _rms(accompaniment)
                    mix = vocal + accompaniment
                    signals = (("vocal", vocal), ("accompaniment", accompaniment), ("mix", mix))
                    peak = max(np.max(np.abs(signal)) for _, signal in signals)
                    scale = 0.90 / max(0.90, peak)
                    directory = output_dir / song_id / clip_id
                    directory.mkdir(parents=True, exist_ok=True)
                    row = {"song_id": song_id, "clip_id": clip_id, "split": split, "text": text}
                    for name, signal in signals:
                        path = directory / f"{name}.wav"
                        write_wav(path, scale * signal, _SAMPLE_RATE)
                        row[name] = path.relative_to(output_dir).as_posix()
                    manifest.write(json.dumps(row) + "\n")
    return sources

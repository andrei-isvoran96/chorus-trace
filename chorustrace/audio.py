"""PCM WAV I/O and NumPy log-mel features."""

import math
import os
import wave
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    window_size: int = 400
    hop_size: int = 160
    fft_size: int = 512
    n_mels: int = 40
    min_hz: float = 50.0

    def __post_init__(self):
        sizes = (self.sample_rate, self.window_size, self.hop_size, self.fft_size, self.n_mels)
        if not all(_positive_integer(size) for size in sizes):
            raise ValueError("sample rate, window, hop, FFT size, and mel count must be positive integers")
        if not 3 <= self.window_size <= self.fft_size or self.hop_size > self.window_size:
            raise ValueError("require 3 <= window size <= FFT size and hop size <= window size")
        if not isinstance(self.min_hz, Real) or not 0 <= self.min_hz < self.sample_rate / 2:
            raise ValueError("minimum frequency must be between zero and Nyquist")


def _positive_integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def read_wav(path):
    """Read integer PCM as mono float64 samples in [-1, 1)."""
    try:
        with wave.open(os.fspath(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            data = wav.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"{path}: expected an integer PCM WAV file ({exc or 'incomplete header'})") from exc

    if width not in (1, 2, 3, 4):
        raise ValueError(f"{path}: expected 8-, 16-, 24- or 32-bit PCM")
    if not sample_rate or not frame_count or len(data) != frame_count * channels * width:
        raise ValueError(f"{path}: empty or truncated PCM WAV")

    if width == 1:
        integers = np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128
    elif width == 3:
        octets = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        integers = (packed ^ (1 << 23)) - (1 << 23)
    else:
        integers = np.frombuffer(data, dtype=f"<i{width}")

    samples = integers.reshape(-1, channels).mean(axis=1, dtype=np.float64)
    return samples / (1 << (8 * width - 1)), sample_rate


def write_wav(path, samples, sample_rate):
    """Write mono or sample-major multichannel float samples as PCM16."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim not in (1, 2) or not samples.size or not np.isfinite(samples).all():
        raise ValueError("expected a nonempty, finite WAV sample array")
    if not _positive_integer(sample_rate):
        raise ValueError("sample rate must be a positive integer")

    channels = 1 if samples.ndim == 1 else samples.shape[1]
    quantized = np.rint(np.clip(samples, -1, 1) * 32768).clip(-32768, 32767).astype("<i2")
    with wave.open(os.fspath(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(quantized.tobytes())


def _resample(samples, source_rate, target_rate):
    if source_rate == target_rate:
        return samples

    ratio = target_rate / source_rate
    if ratio < 1:
        # Band-limit before interpolation when the target Nyquist frequency is lower.
        radius = math.ceil(8 / ratio)
        offsets = np.arange(-radius, radius + 1)
        kernel = np.sinc(0.94 * ratio * offsets) * np.hanning(offsets.size)
        padded = np.pad(samples, radius, mode="edge")
        samples = np.convolve(padded, kernel / kernel.sum(), mode="valid")

    count = max(1, (len(samples) * target_rate + source_rate // 2) // source_rate)
    return np.interp(np.arange(count) / ratio, np.arange(len(samples)), samples)


@lru_cache(maxsize=8)
def _spectral_weights(config):
    mel_edges = np.linspace(
        2595 * np.log10(1 + config.min_hz / 700),
        2595 * np.log10(1 + config.sample_rate / 1400),
        config.n_mels + 2,
    )
    hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
    frequencies = np.fft.rfftfreq(config.fft_size, d=1 / config.sample_rate)[:, None]
    rising = (frequencies - hz_edges[:-2]) / (hz_edges[1:-1] - hz_edges[:-2])
    falling = (hz_edges[2:] - frequencies) / (hz_edges[2:] - hz_edges[1:-1])
    filters = np.maximum(0, np.minimum(rising, falling))
    if (filters.max(axis=0) == 0).any():
        raise ValueError("FFT resolution is too low for the requested mel filters")
    return np.hanning(config.window_size), filters


_DEFAULT_FEATURES = FeatureConfig()


def log_mel(samples, sample_rate, config=_DEFAULT_FEATURES):
    """Return time-major natural-log mel powers for a mono waveform."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("expected nonempty, finite mono audio")
    if not _positive_integer(sample_rate):
        raise ValueError("sample rate must be a positive integer")
    samples = _resample(samples, int(sample_rate), config.sample_rate)

    remaining = max(0, len(samples) - config.window_size)
    frame_count = 1 + (remaining + config.hop_size - 1) // config.hop_size
    padded_length = (frame_count - 1) * config.hop_size + config.window_size
    samples = np.pad(samples, (0, padded_length - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, config.window_size)[::config.hop_size]
    window, filters = _spectral_weights(config)
    spectrum = np.fft.rfft(frames * window, n=config.fft_size, axis=1)
    power = (np.abs(spectrum) / window.sum()) ** 2
    return np.log(np.maximum(power @ filters, 1e-10))


def features_from_wav(path, config=_DEFAULT_FEATURES):
    """Read a PCM clip and extract time-major log-mel features."""
    samples, sample_rate = read_wav(path)
    return log_mel(samples, sample_rate, config)

import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chorustrace.audio import (
    FeatureConfig,
    features_from_wav,
    log_mel,
    read_wav,
    write_wav,
)


class WavTests(unittest.TestCase):
    def test_reads_all_common_integer_pcm_widths(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pcm.wav"
            for width in (1, 2, 3, 4):
                with self.subTest(width=width):
                    scale = 1 << (8 * width - 1)
                    integers = (-scale, -2, 0, 1, scale - 1)
                    if width == 1:
                        data = bytes(value + scale for value in integers)
                    else:
                        data = b"".join(value.to_bytes(width, "little", signed=True) for value in integers)
                    with wave.open(str(path), "wb") as wav:
                        wav.setparams((1, width, 8000, 0, "NONE", "not compressed"))
                        wav.writeframes(data)

                    samples, rate = read_wav(path)
                    self.assertEqual(rate, 8000)
                    np.testing.assert_array_equal(samples, np.array(integers) / scale)

    def test_stereo_downmix_and_partial_frame(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_wav(path, [[0.75, -0.75], [0.5, 0.25], [-1.0, -1.0]], 8000)
            samples, rate = read_wav(path)
            self.assertEqual(rate, 8000)
            np.testing.assert_array_equal(samples, [0.0, 0.375, -1.0])
            self.assertEqual(features_from_wav(path).shape, (1, 40))

    def test_rejects_truncated_pcm(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.wav"
            write_wav(path, [0.0, 0.25, 0.5], 16000)
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "truncated PCM WAV"):
                read_wav(path)


class LogMelTests(unittest.TestCase):
    @staticmethod
    def tone(hz, rate=16000):
        return np.sin(2 * np.pi * hz * np.arange(rate // 4) / rate)

    def test_frequency_order_and_hop(self):
        low = log_mel(self.tone(500), 16000)
        high = log_mel(self.tone(2500), 16000)
        self.assertEqual(low.shape, (24, 40))
        self.assertEqual(low.shape, high.shape)
        self.assertLess(np.argmax(low[5]), np.argmax(high[5]))
        self.assertEqual(log_mel(np.zeros(401), 16000).shape, (2, 40))

    def test_silence_has_finite_log_floor(self):
        features = log_mel([0.0], 16000)
        np.testing.assert_allclose(features, np.log(1e-10))

    def test_downsampling_suppresses_frequencies_above_target_nyquist(self):
        in_band = log_mel(self.tone(3000, 48000), 48000)
        out_of_band = log_mel(self.tone(13000, 48000), 48000)
        in_band_strength = np.exp(in_band[5]).sum()
        self.assertLess(np.exp(out_of_band[5]).sum(), 0.01 * in_band_strength)

    def test_rejects_invalid_feature_config(self):
        with self.assertRaises(ValueError):
            FeatureConfig(fft_size=256)
        with self.assertRaises(ValueError):
            FeatureConfig(hop_size=0)


if __name__ == "__main__":
    unittest.main()

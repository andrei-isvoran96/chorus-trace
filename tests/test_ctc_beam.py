import itertools
import unittest
from collections import defaultdict

import numpy as np

from chorustrace.ctc import greedy_decode
from chorustrace.ctc_beam import CTCPrefixBeam


def _collapse(labels):
    return tuple(label for label, _ in itertools.groupby(labels) if label)


class CTCPrefixBeamTests(unittest.TestCase):
    def test_maximum_prefix_mass_and_alignment_match_brute_enumeration(self):
        rng = np.random.default_rng(21)
        examples = [np.array([[0.40, 0.35, 0.25], [0.40, 0.35, 0.25]])]
        examples.extend(rng.dirichlet(np.ones(3), size=4) for _ in range(20))
        for probabilities in examples:
            frames, classes = probabilities.shape
            masses = defaultdict(float)
            best_paths = defaultdict(float)
            for labels in itertools.product(range(classes), repeat=frames):
                sequence = _collapse(labels)
                probability = float(np.prod(probabilities[np.arange(frames), labels]))
                masses[sequence] += probability
                best_paths[sequence] = max(best_paths[sequence], probability)
            expected = max(masses, key=masses.get)

            beam = CTCPrefixBeam(classes ** frames)
            beam.advance(np.log(probabilities[:1]))
            beam.advance(np.log(probabilities[1:]))
            labels = beam.best_path()
            self.assertEqual(expected, _collapse(labels))
            self.assertAlmostEqual(best_paths[expected], np.prod(probabilities[np.arange(frames), labels]))

        self.assertEqual(greedy_decode(examples[0]), [])
        beam = CTCPrefixBeam(9)
        beam.advance(np.log(examples[0]))
        self.assertEqual(_collapse(beam.best_path()), (1,))

    def test_pruning_preserves_last_token_and_repeated_occurrence_across_batches(self):
        probabilities = np.array([[0.01, 0.98, 0.01], [0.30, 0.32, 0.38],
                                  [0.99, 0.005, 0.005], [0.01, 0.98, 0.01]])
        beam = CTCPrefixBeam(1, token_top_k=1, logit_margin=0.1)
        for row in probabilities:
            beam.advance(np.log(row[None]))
        np.testing.assert_array_equal(beam.best_path(), [1, 1, 0, 1])

        whole = CTCPrefixBeam(1, token_top_k=1, logit_margin=0.1)
        whole.advance(np.log(probabilities))
        np.testing.assert_array_equal(beam.best_path(), whole.best_path())


if __name__ == "__main__":
    unittest.main()

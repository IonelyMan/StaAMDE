import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase2.graph.data import load_train_val_samples, split_indices_auto
from phase2.graph.train import parse_args, training_class_weights


class GraphTrainingDataTests(unittest.TestCase):
    def test_training_defaults(self):
        with patch.object(sys, "argv", ["train", "--output", "unused"]):
            args = parse_args()
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.dropout, 0.25)
        self.assertFalse(hasattr(args, "class_weight"))
        self.assertFalse(hasattr(args, "micro_batch_size"))

    def test_existing_train_val_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, label, name in [
                ("train", "benign", "a"), ("train", "mal", "b"),
                ("val", "benign", "c"), ("val", "mal", "d"),
                ("test", "mal", "e"),
            ]:
                graph = root / split / label / name / "hetero_graph.pt"
                graph.parent.mkdir(parents=True, exist_ok=True)
                graph.touch()
            samples = load_train_val_samples(root, None)
            labels = np.asarray([sample["label"] for sample in samples])
            train, val, test, source = split_indices_auto(labels, samples, 0.2, 0.1, 42)
            self.assertEqual(source, "existing_train_val")
            self.assertEqual((len(train), len(val), len(test)), (2, 2, 0))
            self.assertNotIn("e", {sample["sample_id"] for sample in samples})

    def test_existing_index_split_reserves_test(self):
        samples = [{"split": split} for split in ("train", "train", "val", "test", "test")]
        labels = np.array([0, 1, 0, 1, 1])
        train, val, test, source = split_indices_auto(labels, samples, 0.2, 0.1, 42)
        self.assertEqual(source, "existing")
        self.assertEqual((train, val, test), ([0, 1], [2], [3, 4]))

    def test_class_weights_are_computed_from_training_labels_only(self):
        # The held-out labels deliberately reverse the training imbalance.
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        counts, weights = training_class_weights(labels, [0, 1, 2, 3])
        self.assertEqual(counts, [3, 1])
        np.testing.assert_allclose(weights, [2.0 / 3.0, 2.0])
        self.assertAlmostEqual(sum(c * w for c, w in zip(counts, weights)) / 4, 1.0)

    def test_class_weights_reject_missing_training_class(self):
        with self.assertRaises(ValueError):
            training_class_weights(np.array([0, 0, 1]), [0, 1])


if __name__ == "__main__":
    unittest.main()

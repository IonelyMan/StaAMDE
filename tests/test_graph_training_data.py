import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase2.graph.data import global_node_type_ids, load_train_val_samples, split_indices_auto
from phase2.graph.train import parse_args, training_class_weights


class GraphTrainingDataTests(unittest.TestCase):
    def test_training_defaults(self):
        with patch.object(sys, "argv", ["train", "--output", "unused"]):
            args = parse_args()
        self.assertEqual(args.readout, "mean")
        self.assertEqual(args.class_weight, "balanced")
        self.assertEqual(args.lr_scheduler, "none")
        self.assertEqual(args.grad_clip, 0.0)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.dropout, 0.25)

    def test_existing_splits_exclude_test_and_class_weights_use_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, label, name in [
                ("train", "benign", "a"),
                ("train", "benign", "b"),
                ("train", "mal", "c"),
                ("val", "benign", "d"),
                ("val", "mal", "e"),
                ("test", "mal", "f"),
            ]:
                graph = root / split / label / name / "hetero_graph.pt"
                graph.parent.mkdir(parents=True, exist_ok=True)
                graph.touch()
            samples = load_train_val_samples(root, None)
            labels = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
            train_idx, val_idx, test_idx, source = split_indices_auto(labels, samples, 0.2, 0.1, 42)
            self.assertEqual(source, "existing_train_val")
            self.assertEqual((len(train_idx), len(val_idx), len(test_idx)), (3, 2, 0))
            self.assertNotIn("f", {sample["sample_id"] for sample in samples})
            counts, weights = training_class_weights(labels, train_idx, "balanced")
            self.assertEqual(counts, [2, 1])
            self.assertAlmostEqual(weights[1] / weights[0], 2.0)
            self.assertAlmostEqual((counts[0] * weights[0] + counts[1] * weights[1]) / 3, 1.0)

    def test_partial_existing_splits_cannot_be_randomly_reassigned(self):
        samples = [{"split": "train"}, {"split": "test"}]
        with self.assertRaisesRegex(ValueError, "拒绝重新随机划分"):
            split_indices_auto(np.array([0, 1]), samples, 0.2, 0.1, 42)

    def test_index_with_all_splits_reserves_test_and_excludes_it_from_weights(self):
        samples = [{"split": split} for split in ("train", "train", "val", "test", "test")]
        labels = np.array([0, 1, 0, 1, 1])
        train_idx, val_idx, test_idx, source = split_indices_auto(labels, samples, 0.2, 0.1, 42)
        self.assertEqual(source, "existing")
        self.assertEqual((train_idx, val_idx, test_idx), ([0, 1], [2], [3, 4]))
        counts, weights = training_class_weights(labels, train_idx, "balanced")
        self.assertEqual(counts, [1, 1])
        self.assertEqual(weights, [1.0, 1.0])

    def test_node_types_keep_global_ids_when_local_types_differ(self):
        vocabulary = ["api", "app", "method", "permission"]
        self.assertEqual(global_node_type_ids(["app", "permission"], vocabulary), [1, 3])
        self.assertEqual(global_node_type_ids(["app", "api", "other"], vocabulary), [1, 0, 4])


if __name__ == "__main__":
    unittest.main()

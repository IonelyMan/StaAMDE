import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ensemble_predict


class EnsembleLeakageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "out"
        self.test_static = self.write("test_static", "test", [0.9, 0.1])
        self.test_graph = self.write("test_graph", "test", [0.1, 0.9])
        self.val_static = self.write("val_static", "val", [0.1, 0.9])
        self.val_graph = self.write("val_graph", "val", [0.9, 0.1])

    def write(self, name, split, probs, ids=None, labels=None, omit=()):
        ids = ids or [f"{split}-negative", f"{split}-positive"]
        labels = labels or [0, 1]
        fields = [col for col in ("sample_id", "split", "y_true", "prob_malware") if col not in omit]
        path = self.root / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for sid, label, prob in zip(ids, labels, probs):
                writer.writerow(dict(sample_id=sid, split=split, y_true=label, prob_malware=prob))
        return path

    def run_main(self, *extra):
        args = ["ensemble_predict.py", "--input", f"static={self.test_static}",
                "--input", f"graph={self.test_graph}", "--output", str(self.output), *extra]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            ensemble_predict.main()

    def tuning_args(self, static=None, graph=None):
        return ["--tune-input", f"static={static or self.val_static}",
                "--tune-input", f"graph={graph or self.val_graph}"]

    def test_tuning_rejects_missing_validation_files(self):
        for flag in ("--tune-weights", "--tune-threshold"):
            with self.subTest(flag=flag), self.assertRaises((ValueError, SystemExit)):
                self.run_main(flag)
            self.assertFalse(self.output.exists())

    def test_rejects_test_predictions_as_validation(self):
        with self.assertRaises((ValueError, SystemExit)):
            self.run_main("--tune-weights", *self.tuning_args(self.test_static, self.test_graph))
        self.assertFalse(self.output.exists())

    def test_rejects_missing_validation_split(self):
        missing = self.write("val_missing", "val", [0.1, 0.9], omit=("split",))
        with self.assertRaises((ValueError, SystemExit)):
            self.run_main("--tune-weights", *self.tuning_args(missing, self.val_graph))

    def test_allows_same_sample_id_in_validation_and_test(self):
        ids = ["test-negative", "val-positive"]
        val_static = self.write("overlap_static", "val", [0.1, 0.9], ids=ids)
        val_graph = self.write("overlap_graph", "val", [0.1, 0.9], ids=ids)
        self.run_main("--tune-threshold", *self.tuning_args(val_static, val_graph))
        metrics = json.loads((self.output / "ensemble_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["tuning"]["n_samples"], 2)

    def test_unmatched_validation_sample_id_is_excluded(self):
        ids = ["test-negative", "val-negative", "val-positive"]
        val_static = self.write("partial_overlap", "val", [0.2, 0.1, 0.9],
                                ids=ids, labels=[0, 0, 1])
        self.run_main("--tune-threshold", *self.tuning_args(val_static, self.val_graph))
        metrics = json.loads((self.output / "ensemble_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["tuning"]["n_samples"], 2)

    def test_rejects_when_no_sample_key_can_align(self):
        graph = self.write("conflict", "val", [0.9, 0.1], labels=[1, 0])
        with self.assertRaisesRegex(ValueError, "没有可对齐的 sample_key"):
            self.run_main("--tune-weights", *self.tuning_args(self.val_static, graph))

    def test_same_id_with_different_labels_within_one_split_can_align(self):
        static = self.write("two_labels_static", "val", [0.1, 0.9], ids=["same", "same"])
        graph = self.write("two_labels_graph", "val", [0.2, 0.8], ids=["same", "same"])
        merged, _ = ensemble_predict.merge_predictions(
            [f"static={static}", f"graph={graph}"], required_split="val")
        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged["y_true"]), {0, 1})

    def test_conflicting_label_for_same_id_only_excludes_that_key(self):
        ids = ["same", "benign", "mal"]
        static = self.write("partial_conflict_static", "val", [0.1, 0.2, 0.8],
                            ids=ids, labels=[0, 0, 1])
        graph = self.write("partial_conflict_graph", "val", [0.9, 0.3, 0.7],
                           ids=ids, labels=[1, 0, 1])
        self.run_main("--tune-weights", *self.tuning_args(static, graph))
        metrics = json.loads((self.output / "ensemble_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["tuning"]["n_samples"], 2)

    def test_tunes_on_validation_and_outputs_only_test(self):
        self.run_main("--tune-weights", "--tune-threshold", *self.tuning_args())
        metrics = json.loads((self.output / "ensemble_metrics.json").read_text(encoding="utf-8"))
        self.assertGreater(metrics["weights"]["static"], 0.5)
        static_weight = metrics["weights"]["static"]
        graph_weight = metrics["weights"]["graph"]
        self.assertGreater(metrics["threshold"], static_weight * 0.1 + graph_weight * 0.9)
        self.assertLessEqual(metrics["threshold"], static_weight * 0.9 + graph_weight * 0.1)
        with (self.output / "ensemble_predictions.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({row["sample_id"] for row in rows}, {"test-negative", "test-positive"})
        self.assertEqual({row["split"] for row in rows}, {"test"})

    def test_fixed_parameters_need_no_validation_file(self):
        self.run_main()
        metrics = json.loads((self.output / "ensemble_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["weights"], {"static": 0.5, "graph": 0.5})
        self.assertEqual(metrics["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()

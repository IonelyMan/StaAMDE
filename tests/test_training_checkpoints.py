import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phase2.lightgbm.model import load_artifacts
from phase2.lightgbm.train import best_checkpoint_callback


class FakeBooster:
    def model_to_string(self, num_iteration):
        return f"trees={num_iteration}"


class TrainingCheckpointTests(unittest.TestCase):
    def test_lightgbm_best_round_survives_later_failure(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            callback = best_checkpoint_callback(root, {"vectorizer": True}, np.array([0]), ["feature"])
            for iteration, score in [(0, 0.6), (1, 0.4), (2, 0.7)]:
                callback(SimpleNamespace(iteration=iteration, model=FakeBooster(),
                                         evaluation_result_list=[("valid_0", "pr_auc", score, True)]))
            checkpoint = root / "best_checkpoint.joblib"
            before = checkpoint.read_bytes()
            class FailingBooster:
                def model_to_string(self, num_iteration):
                    raise RuntimeError("simulated out of memory")

            with self.assertRaises(RuntimeError):
                callback(SimpleNamespace(iteration=3, model=FailingBooster(),
                                         evaluation_result_list=[("valid_0", "pr_auc", 0.8, True)]))
            self.assertEqual(checkpoint.read_bytes(), before)
            model, vectorizer, indices, names = load_artifacts(root)
            self.assertEqual(model.model_string, "trees=3")
            self.assertEqual(vectorizer, {"vectorizer": True})
            self.assertEqual(indices.tolist(), [0])
            self.assertEqual(names, ["feature"])

if __name__ == "__main__":
    unittest.main()

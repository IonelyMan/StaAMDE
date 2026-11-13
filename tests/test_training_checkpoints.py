import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from phase2.graph.train import save_best_checkpoint
from phase2.lightgbm.model import load_artifacts
from phase2.lightgbm.train import best_checkpoint_callback


class FakeBooster:
    def model_to_string(self, num_iteration):
        return f"trees={num_iteration}"


class PickleTorch:
    @staticmethod
    def save(payload, path):
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)


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

    def test_graph_best_checkpoint_is_not_replaced_by_failed_save(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "gatv2_model.pt"
            args = SimpleNamespace(model_type="atgatv2", hidden_channels=8, layers=2, heads=2,
                                   dropout=0.1, type_embedding_dim=4, readout="mean")
            save_best_checkpoint(PickleTorch, path, {"weight": 1}, (["api"], []), args)
            original = path.read_bytes()

            class FailingTorch:
                @staticmethod
                def save(payload, destination):
                    raise RuntimeError("simulated out of memory while saving")

            with self.assertRaises(RuntimeError):
                save_best_checkpoint(FailingTorch, path, {"weight": 2}, (["api"], []), args)
            self.assertEqual(path.read_bytes(), original)
            with path.open("rb") as handle:
                saved = pickle.load(handle)
                self.assertEqual(saved["state_dict"], {"weight": 1})
                self.assertEqual(saved["model_config"]["node_type_order"], "legacy")


if __name__ == "__main__":
    unittest.main()

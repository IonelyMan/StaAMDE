import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deduplicate import find_missing_graph_edges as scanner


class FindMissingGraphEdgesTests(unittest.TestCase):
    def test_scans_split_and_uses_preferred_graph_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, name in (("train", "bad"), ("train", "good"), ("val", "other")):
                directory = root / split / "mal" / name
                directory.mkdir(parents=True)
                (directory / "hetero_graph.pt").touch()
            (root / "train" / "mal" / "bad" / "hetero_graph_data.pt").touch()

            def load(path):
                return SimpleNamespace(edge_index=None if path.parent.name == "bad" else object())

            with patch.object(scanner, "load_graph", side_effect=load) as loader, \
                 patch.object(scanner, "to_homogeneous_graph", side_effect=lambda graph: graph):
                targets, count = scanner.scan(root, "train")
            self.assertEqual(count, 2)
            self.assertEqual([(t.split, t.label, t.apk_name) for t in targets], [("train", "mal", "bad.apk")])
            self.assertEqual(loader.call_args_list[0].args[0].name, "hetero_graph_data.pt")

    def test_failure_does_not_create_partial_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "train" / "mal" / "bad" / "hetero_graph.pt"
            graph.parent.mkdir(parents=True)
            graph.touch()
            output = root / "manifest.csv"
            with patch.object(sys, "argv", ["scanner", "--graph-root", str(root), "--output", str(output)]), \
                 patch.object(scanner, "load_graph", side_effect=OSError("corrupt")):
                with self.assertRaisesRegex(SystemExit, "无法加载或转换图"):
                    scanner.main()
            self.assertFalse(output.exists())

    def test_rejects_symlinked_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "train" / "mal"
            parent.mkdir(parents=True)
            target = root / "outside"
            target.mkdir()
            try:
                (parent / "linked").symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "符号链接"):
                scanner.scan(root, "train")

    def test_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "train" / "benign" / "bad" / "hetero_graph.pt"
            graph.parent.mkdir(parents=True)
            graph.touch()
            output = root / "manifest.csv"
            with patch.object(sys, "argv", ["scanner", "--graph-root", str(root), "--output", str(output)]), \
                 patch.object(scanner, "load_graph", return_value=object()), \
                 patch.object(scanner, "to_homogeneous_graph", return_value=SimpleNamespace(edge_index=None)):
                scanner.main()
                with output.open(encoding="utf-8-sig", newline="") as handle:
                    self.assertEqual(list(csv.DictReader(handle)), [dict(split="train", label="benign", apk_name="bad.apk")])
                with self.assertRaises(SystemExit):
                    scanner.main()
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

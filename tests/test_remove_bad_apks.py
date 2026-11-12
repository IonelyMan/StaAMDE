import csv
import io
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from deduplicate.remove_bad_apks import main, target_from_fields


class RemoveBadApksTest(unittest.TestCase):
    def test_rejects_path_in_sample_name(self):
        with self.assertRaises(SystemExit):
            target_from_fields("test", "0", "../other.apk")

    def test_preview_and_cross_split_apply(self):
        workspace = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="remove_bad_apks_", dir=workspace) as temporary:
            root = Path(temporary)
            apk_root, static_root, graph_root = (root / name for name in ("apks", "static", "graphs"))
            for data_root in (apk_root, static_root, graph_root):
                data_root.mkdir()
            samples = [
                ("test", "benign", "bad.apk"),
                ("val", "mal", "bad.apk"),
                ("train", "benign", "bad.apk"),
                ("test", "benign", "keep.apk"),
            ]
            for split, label, name in samples:
                apk = apk_root / split / label / name
                apk.parent.mkdir(parents=True, exist_ok=True)
                apk.write_bytes(b"apk")
                for data_root, filename in ((static_root, "static_features_report.json"),
                                            (graph_root, "hetero_graph.pt")):
                    output = data_root / split / label / name[:-4] / filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"data")
            for split in ("test", "val", "train"):
                rows = [{"apk_name": name, "label": "0" if label == "benign" else "1", "split": split}
                        for row_split, label, name in samples if row_split == split]
                index = graph_root / split / "index.csv"
                with index.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["apk_name", "label", "split"])
                    writer.writeheader()
                    writer.writerows(rows)

            manifest = root / "bad.csv"
            manifest.write_text("split,label,apk_name\ntest,benign,bad.apk\nval,1,bad\n", encoding="utf-8")
            command = ["remove_bad_apks", "--apk-root", str(apk_root), "--static-root", str(static_root),
                       "--graph-root", str(graph_root), "--manifest", str(manifest),
                       "--audit-dir", str(root / "audit")]
            with patch("sys.argv", command), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((apk_root / "test" / "benign" / "bad.apk").exists())
            self.assertFalse((root / "audit").exists())

            with patch("sys.argv", command + ["--apply"]), redirect_stdout(io.StringIO()):
                main()
            for split, label in (("test", "benign"), ("val", "mal")):
                self.assertFalse((apk_root / split / label / "bad.apk").exists())
                self.assertFalse((static_root / split / label / "bad").exists())
                self.assertFalse((graph_root / split / label / "bad").exists())
            self.assertTrue((apk_root / "train" / "benign" / "bad.apk").exists())
            self.assertTrue((apk_root / "test" / "benign" / "keep.apk").exists())
            with (graph_root / "test" / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual([row["apk_name"] for row in csv.DictReader(handle)], ["keep.apk"])
            with (graph_root / "val" / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            self.assertTrue((root / "audit" / "test_index_before.csv").exists())
            self.assertTrue((root / "audit" / "val_index_before.csv").exists())

    def test_automatic_detection_across_splits(self):
        with tempfile.TemporaryDirectory(prefix="auto_bad_apks_", dir=Path.cwd()) as temporary:
            root = Path(temporary)
            apk_root, static_root, graph_root = (root / name for name in ("apks", "static", "graphs"))
            for data_root in (apk_root, static_root, graph_root):
                data_root.mkdir()
            samples = [
                ("test", "benign", "static_bad.apk", "static"),
                ("val", "mal", "graph_bad.apk", "graph"),
                ("train", "benign", "invalid.apk", "zip"),
                ("test", "benign", "keep.apk", "ok"),
                ("val", "benign", "unprocessed.apk", "unprocessed"),
            ]
            for split, label, name, kind in samples:
                apk = apk_root / split / label / name
                apk.parent.mkdir(parents=True, exist_ok=True)
                if kind == "zip":
                    apk.write_bytes(b"not an APK archive")
                else:
                    with zipfile.ZipFile(apk, "w") as archive:
                        archive.writestr("AndroidManifest.xml", "manifest")
                if kind != "unprocessed":
                    report = static_root / split / label / name[:-4] / "static_features_report.json"
                    report.parent.mkdir(parents=True, exist_ok=True)
                    report.write_text('{"error":"analysis failed"}' if kind == "static" else "{}", encoding="utf-8")
                    graph = graph_root / split / label / name[:-4] / "hetero_graph.pt"
                    graph.parent.mkdir(parents=True, exist_ok=True)
                    graph.write_bytes(b"graph")
            for split in ("train", "val", "test"):
                rows = [{"apk_name": name, "label": "0" if label == "benign" else "1",
                         "split": split, "status": "failed" if kind == "graph" else "ok", "error": ""}
                        for row_split, label, name, kind in samples if row_split == split and kind != "unprocessed"]
                index = graph_root / split / "index.csv"
                with index.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["apk_name", "label", "split", "status", "error"])
                    writer.writeheader()
                    writer.writerows(rows)
            command = ["remove_bad_apks", "--apk-root", str(apk_root), "--static-root", str(static_root),
                       "--graph-root", str(graph_root), "--audit-dir", str(root / "audit")]
            with patch("sys.argv", command), redirect_stdout(io.StringIO()) as preview:
                main()
            self.assertIn("静态提取失败", preview.getvalue())
            self.assertIn("图构建失败", preview.getvalue())
            self.assertIn("APK 不是有效 ZIP", preview.getvalue())
            self.assertTrue((apk_root / "test" / "benign" / "static_bad.apk").exists())
            with patch("sys.argv", command + ["--split", "val"]), redirect_stdout(io.StringIO()) as scoped:
                main()
            self.assertIn("graph_bad.apk", scoped.getvalue())
            self.assertNotIn("static_bad.apk", scoped.getvalue())
            self.assertNotIn("invalid.apk", scoped.getvalue())
            with patch("sys.argv", command + ["--apply"]), redirect_stdout(io.StringIO()):
                main()
            for split, label, name, kind in samples:
                exists = kind in {"ok", "unprocessed"}
                self.assertEqual((apk_root / split / label / name).exists(), exists)
                if kind != "unprocessed":
                    self.assertEqual((static_root / split / label / name[:-4]).exists(), exists)
                    self.assertEqual((graph_root / split / label / name[:-4]).exists(), exists)
            with (graph_root / "test" / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual([row["apk_name"] for row in csv.DictReader(handle)], ["keep.apk"])


if __name__ == "__main__":
    unittest.main()

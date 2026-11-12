import csv
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from deduplicate.audit_extraction import audit_split


class AuditExtractionTests(unittest.TestCase):
    def test_reports_exact_sample_and_failure_reason(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            apk_root, static_root, graph_root = (root / name for name in ("apks", "static", "graphs"))
            for label, name, error in (("benign", "bad.apk", "Analysis Failed"),
                                       ("mal", "good.apk", "")):
                apk = apk_root / "train" / label / name
                apk.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(apk, "w") as archive:
                    archive.writestr("AndroidManifest.xml", "ok")
                report = static_root / "train" / label / name[:-4] / "static_features_report.json"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps({"error": error} if error else {}), encoding="utf-8")
                graph = graph_root / "train" / label / name[:-4] / "hetero_graph.pt"
                graph.parent.mkdir(parents=True, exist_ok=True)
                graph.write_bytes(b"graph")
            index = graph_root / "train" / "index.csv"
            with index.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["apk_name", "label", "status", "error"])
                writer.writeheader()
                writer.writerows([
                    {"apk_name": "bad.apk", "label": "0", "status": "failed", "error": "graph failed"},
                    {"apk_name": "good.apk", "label": "1", "status": "ok", "error": ""},
                ])
            output = io.StringIO()
            with redirect_stdout(output):
                count = audit_split(apk_root, static_root, graph_root, "train")
            self.assertEqual(count, 1)
            self.assertIn("bad: 静态报告含 error 字段（加载器会跳过）: Analysis Failed；图索引状态或错误: graph failed", output.getvalue())
            self.assertNotIn("good: ", output.getvalue())
            self.assertTrue((apk_root / "train" / "benign" / "bad.apk").exists())


if __name__ == "__main__":
    unittest.main()

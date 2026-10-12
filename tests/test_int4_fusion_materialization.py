from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from scripts.materialize_edgellm_int4_fusion import copy_operation, materialize


class Int4FusionMaterializationTests(unittest.TestCase):
    def test_group_scale_planes_are_interleaved_by_row_not_whole_tensor(self):
        # Two projections, three quantization groups: Q=[[1,2],[3,4],[5,6]], K=[[7],[8],[9]].
        source = BytesIO(bytes([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        target = BytesIO()
        operation = {"mode": "row_concat", "rows": 3, "offset": 0, "length": 9,
                     "sources": [{"offset": 0, "length": 6}, {"offset": 6, "length": 3}]}
        self.assertEqual(copy_operation(source, target, operation, 9), 9)
        self.assertEqual(target.getvalue(), bytes([1, 2, 7, 3, 4, 8, 5, 6, 9]))

    def test_n_major_packed_weight_blocks_preserve_projection_bytes(self):
        source = BytesIO(bytes(range(32)))
        target = BytesIO()
        operation = {"mode": "concat", "rows": 1, "offset": 0, "length": 24,
                     "sources": [{"offset": 16, "length": 16}, {"offset": 0, "length": 8}]}
        copy_operation(source, target, operation, 32)
        self.assertEqual(target.getvalue()[:16], bytes(range(16, 32)))
        self.assertEqual(target.getvalue()[16:], bytes(range(8)))

    def test_invalid_range_and_partial_row_are_rejected(self):
        for offset, length, rows in ((-1, 2, 1), (0, 5, 1), (0, 3, 2)):
            operation = {"mode": "row_concat", "rows": rows, "offset": 0, "length": length,
                         "sources": [{"offset": offset, "length": length}]}
            with self.subTest(offset=offset, length=length, rows=rows):
                with self.assertRaises(ValueError):
                    copy_operation(BytesIO(b"1234"), BytesIO(), operation, 4)

    def test_materialization_pins_sources_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.data"
            source.write_bytes(b"12345678")
            prepared = root / "prepared"
            prepared.mkdir()
            graph = b"prepared graph bytes"
            (prepared / "model.onnx").write_bytes(graph)
            plan = {"schema_version": "parksight_edgellm_int4_fusion_plan_v1",
                    "layout": "edgellm_interleave4_n_major_group128", "source_graph_sha256": "a" * 64,
                    "source_weights": {"size_bytes": 8, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                    "candidate_graph_sha256": hashlib.sha256(graph).hexdigest(), "fusion_weights_bytes": 8,
                    "groups": [{"layer": 0}], "operations": [{"mode": "concat", "rows": 1,
                    "offset": 0, "length": 8, "sources": [{"offset": 0, "length": 8}]}]}
            (prepared / "plan.json").write_text(json.dumps(plan))
            output = root / "candidate"
            result = materialize(prepared, source, output)
            self.assertEqual(result["status"], "materialized")
            self.assertEqual((output / "fusion_weights.data").read_bytes(), source.read_bytes())
            self.assertTrue((output / "model.onnx.data").is_symlink())
            with self.assertRaisesRegex(ValueError, "already exists"):
                materialize(prepared, source, output)
            source.write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "pinned plan"):
                materialize(prepared, source, root / "rejected")
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()

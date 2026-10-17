from __future__ import annotations

import json
import mmap
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.diagnostics.inspect_tensorrt_engine import inspect_engine


class EngineInspectorMemoryTests(unittest.TestCase):
    def test_passes_readonly_mapping_and_keeps_it_open_during_inspection(self):
        mappings = []

        class Logger:
            WARNING = 1

            def __init__(self, level):
                pass

        class Runtime:
            def __init__(self, logger):
                pass

            def deserialize_cuda_engine(self, data):
                self_outer.assertIsInstance(data, mmap.mmap)
                self_outer.assertEqual(data[:], b"engine bytes")
                with self_outer.assertRaises(TypeError):
                    data[0] = 0
                mappings.append(data)

                def layer_info(*args):
                    self_outer.assertFalse(data.closed)
                    return json.dumps({"Name": "node_linear"})

                inspector = SimpleNamespace(get_engine_information=layer_info,
                                            get_layer_information=layer_info)
                return SimpleNamespace(num_layers=1, num_io_tensors=2,
                                       create_engine_inspector=lambda: inspector)

        self_outer = self
        fake = SimpleNamespace(Logger=Logger, Runtime=Runtime, __version__="10.3-test",
                               init_libnvinfer_plugins=lambda *args: None,
                               LayerInformationFormat=SimpleNamespace(JSON=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.engine"
            path.write_bytes(b"engine bytes")
            with patch.dict(sys.modules, {"tensorrt": fake}):
                result = inspect_engine(path)
            self.assertEqual(path.read_bytes(), b"engine bytes")
        self.assertTrue(mappings[0].closed)
        self.assertEqual(result["engine"]["loading"], "read_only_mmap")
        self.assertEqual(result["layers"][0]["Name"], "node_linear")


if __name__ == "__main__":
    unittest.main()

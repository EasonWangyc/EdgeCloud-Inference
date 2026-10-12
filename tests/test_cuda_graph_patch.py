"""Verify Graph capture default, opt-out, and source identity without CUDA."""

import hashlib
import os
import unittest
from unittest.mock import Mock, patch

from scripts.prepare_edgellm_cuda_graph import cuda_graph_source


SOURCE = b'''import os
class LLM:
    def load(self):
        self._runtime.capture_decoding_cuda_graph()
'''


class CudaGraphPatchTests(unittest.TestCase):
    def test_capture_default_and_explicit_disable_preserve_other_values(self):
        updated = cuda_graph_source(SOURCE, hashlib.sha256(SOURCE).hexdigest())
        namespace = {"logger": Mock()}
        exec(compile(updated, "isolated_engine.py", "exec"), namespace)
        for option, expected_calls in [(None, 1), ("0", 1), ("1", 0)]:
            with self.subTest(option=option), patch.dict(os.environ, {}, clear=True):
                if option is not None:
                    os.environ["EDGELLM_DISABLE_CUDA_GRAPH"] = option
                llm = namespace["LLM"]()
                llm._runtime = Mock()
                llm.load()
                self.assertEqual(llm._runtime.capture_decoding_cuda_graph.call_count, expected_calls)

    def test_source_mismatch_duplicate_capture_and_repeat_patch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cuda_graph_source(SOURCE, "0" * 64)
        duplicate = SOURCE + b"        self._runtime.capture_decoding_cuda_graph()\n"
        with self.assertRaisesRegex(ValueError, "layout"):
            cuda_graph_source(duplicate, hashlib.sha256(duplicate).hexdigest())
        updated = cuda_graph_source(SOURCE, hashlib.sha256(SOURCE).hexdigest())
        with self.assertRaisesRegex(ValueError, "already exists"):
            cuda_graph_source(updated, hashlib.sha256(updated).hexdigest())

    def test_crlf_and_unchanged_prefix_are_preserved(self):
        source = SOURCE.replace(b"\n", b"\r\n")
        updated = cuda_graph_source(source, hashlib.sha256(source).hexdigest())
        self.assertTrue(updated.startswith(b"import os\r\nclass LLM:\r\n    def load(self):\r\n"))
        self.assertNotIn(b"\n", updated.replace(b"\r\n", b""))

"""Effective cache/Graph evidence must not persist credentials."""

import unittest

from parksight_vlm.studies.server_config import extract_server_config


class ServerConfigTests(unittest.TestCase):
    def test_capture_actual_cache_and_graph_settings_without_credentials(self):
        payload = {"vllm_config": {
            "model_config": {"revision": "frozen", "dtype": "bfloat16", "hf_token": "private",
                             "multimodal_config": {"mm_processor_cache_gb": 0, "unknown_secret": "private",
                                                   "mm_processor_kwargs": {"min_pixels": 200704, "secret": "private"}}},
            "cache_config": {"enable_prefix_caching": False},
            "scheduler_config": {"max_num_seqs": 4},
            "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4], "cudagraph_mode": [2, 0]},
            "parallel_config": {"tensor_parallel_size": 1},
            "credentials": "private",
        }}
        result = extract_server_config(payload)
        self.assertFalse(result["cache_config"]["enable_prefix_caching"])
        self.assertEqual(result["multimodal_config"]["mm_processor_cache_gb"], 0)
        self.assertEqual(result["model_config"]["revision"], "frozen")
        self.assertEqual(result["multimodal_config"]["mm_processor_kwargs"], {"min_pixels": 200704})
        self.assertEqual(result["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4])
        self.assertNotIn("private", str(result))

    def test_missing_fields_do_not_invent_cache_defaults(self):
        payload = {"vllm_config": {key: {} for key in (
            "model_config", "cache_config", "scheduler_config", "compilation_config", "parallel_config",
        )}}
        result = extract_server_config(payload)
        self.assertEqual(result["cache_config"], {})
        self.assertNotIn("multimodal_config", result)

    def test_text_format_or_incomplete_config_is_rejected(self):
        for payload in ([], {}, {"vllm_config": "ModelConfig(...)"}, {"vllm_config": {}},
                        {"vllm_config": {"model_config": None}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                extract_server_config(payload)

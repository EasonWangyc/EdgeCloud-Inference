"""服务端累积指标窗口差分，避免将进程历史计数当成当前实验。"""

import unittest

from parksight_vlm.studies.server_metrics import summarize_window


class ServerMetricsTests(unittest.TestCase):
    def test_window_uses_deltas_and_preserves_histogram_buckets(self) -> None:
        before = '''# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="a b",engine="0"} 100
vllm:request_success_total{model_name="a b",engine="0"} 4
vllm:time_to_first_token_seconds_count{engine="0"} 4
vllm:time_to_first_token_seconds_sum{engine="0"} 0.2
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 4
vllm:num_requests_running{engine="0"} 0
'''
        after = before.replace(' 100\n', ' 140\n').replace(' 4\n', ' 6\n').replace(' 0.2\n', ' 0.3\n')
        result = summarize_window(before, after, 2.0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["requests_per_second"], 1.0)
        self.assertEqual(result["output_tokens_per_second"], 20.0)
        self.assertAlmostEqual(result["timings"]["time_to_first_token_seconds"]["mean_ms"], 50.0)
        self.assertEqual(result["counter_and_histogram_deltas"]['vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"}'], 2)
        self.assertNotIn('vllm:num_requests_running{engine="0"}', result["counter_and_histogram_deltas"])

    def test_counter_reset_invalidates_window(self) -> None:
        result = summarize_window('vllm:generation_tokens_total 10', 'vllm:generation_tokens_total 1', 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "counter_reset_or_server_restart")

    def test_disappearing_counter_series_invalidates_window(self) -> None:
        before = 'vllm:generation_tokens_total{engine="0"} 10\nvllm:request_success_total{engine="0"} 1'
        after = 'vllm:request_success_total{engine="0"} 2'
        result = summarize_window(before, after, 1)
        self.assertFalse(result["valid"])
        self.assertIn('vllm:generation_tokens_total{engine="0"}', result["reset_samples"])

    def test_label_reordering_does_not_look_like_a_counter_reset(self) -> None:
        before = (
            'vllm:request_success_total{model_name="a,b",engine="0"} 10\n'
            'vllm:generation_tokens_total{model_name="a\\\"b",engine="0"} 100\n'
        )
        after = (
            'vllm:request_success_total{engine="0",model_name="a,b"} 12\n'
            'vllm:generation_tokens_total{engine="0",model_name="a\\\"b"} 140\n'
        )
        result = summarize_window(before, after, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["request_success_count"], 2)
        self.assertEqual(result["generated_tokens"], 40)

    def test_finish_reasons_are_counted_separately(self) -> None:
        before = (
            'vllm:request_success_total{engine="0",finished_reason="stop"} 10\n'
            'vllm:request_success_total{engine="0",finished_reason="length"} 0\n'
            'vllm:generation_tokens_total{engine="0"} 100\n'
        )
        after = (
            'vllm:request_success_total{finished_reason="stop",engine="0"} 11\n'
            'vllm:request_success_total{finished_reason="length",engine="0"} 1\n'
            'vllm:generation_tokens_total{engine="0"} 120\n'
        )
        result = summarize_window(before, after, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["request_success_count"], 2)
        self.assertEqual(result["finish_reason_counts"], {"stop": 1, "length": 1})
        self.assertTrue(result["finish_reason_coverage_complete"])
        self.assertEqual(result["non_stop_request_count"], 1)

    def test_partial_finish_reason_coverage_is_explicit(self) -> None:
        before = 'vllm:request_success_total{finished_reason="stop"} 10\n'
        after = (
            'vllm:request_success_total{finished_reason="stop"} 11\n'
            'vllm:request_success_total{engine="other"} 1\n'
        )
        result = summarize_window(before, after, 1)
        self.assertFalse(result["finish_reason_coverage_complete"])
        self.assertEqual(result["non_stop_request_count"], 0)

    def test_duplicate_series_with_reordered_labels_invalidates_window(self) -> None:
        after = (
            'vllm:request_success_total{engine="0",finished_reason="stop"} 1\n'
            'vllm:request_success_total{finished_reason="stop",engine="0"} 2\n'
        )
        result = summarize_window('', after, 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "duplicate_metric_series")

    def test_process_restart_invalidates_window_even_if_counters_increase(self) -> None:
        before = ('process_start_time_seconds 100\n'
                  'vllm:request_success_total 10\n'
                  'vllm:generation_tokens_total 100\n')
        after = ('process_start_time_seconds 200\n'
                 'vllm:request_success_total 11\n'
                 'vllm:generation_tokens_total 120\n')
        result = summarize_window(before, after, 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "server_process_changed_or_identity_missing")

        result = summarize_window(before, after.replace('process_start_time_seconds 200\n', ''), 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "server_process_changed_or_identity_missing")

    def test_missing_metrics_do_not_become_zero_latency(self) -> None:
        result = summarize_window('', 'vllm:generation_tokens_total 20', 1)
        self.assertIsNone(result["timings"]["inter_token_latency_seconds"]["mean_ms"])
        self.assertIsNone(result["requests_per_second"])

    def test_invalid_window_is_rejected(self) -> None:
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize_window('', '', value)

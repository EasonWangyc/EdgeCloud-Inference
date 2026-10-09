"""Budget queries preserve zero semantics and never create an execution context."""

import unittest

from scripts.inspect_edgellm_weight_streaming import inspect_budget_rows


class Engine:
    streamable_weights_size = 1000
    num_optimization_profiles = 2
    weight_streaming_budget_v2 = 0

    @property
    def weight_streaming_scratch_memory_size(self):
        return 200 if self.weight_streaming_budget_v2 < 500 else 20

    @property
    def device_memory_size_v2(self):
        return self.weight_streaming_scratch_memory_size + 50

    def get_device_memory_size_for_profile_v2(self, profile):
        return self.device_memory_size_v2 - profile * 10

    def create_execution_context(self):
        raise AssertionError("query must not create a context")


class WeightStreamingInspectionTests(unittest.TestCase):
    def test_zero_budget_and_actual_context_include_scratch(self):
        rows = inspect_budget_rows(Engine(), [0, 500, 1000])
        self.assertEqual(rows[0]["actual_budget_bytes"], 0)
        self.assertEqual([r["scratch_bytes"] for r in rows], [200, 20, 20])
        self.assertEqual(rows[1]["profile_context_bytes"], [70, 60])
        self.assertTrue(all(row["budget_applied"] for row in rows))

    def test_silent_budget_rejection_is_recorded_and_partial_rows_are_retained(self):
        class RejectingEngine(Engine):
            _budget = 0

            @property
            def weight_streaming_budget_v2(self):
                return self._budget

            @weight_streaming_budget_v2.setter
            def weight_streaming_budget_v2(self, budget):
                if budget < self.streamable_weights_size:
                    self._budget = budget

        saved = []
        rows = inspect_budget_rows(RejectingEngine(), [0, 500, 1000], on_row=saved.append)
        self.assertEqual(saved, rows)
        self.assertFalse(rows[-1]["budget_applied"])
        self.assertEqual(rows[-1]["actual_budget_bytes"], 500)

        class ThrowingEngine(RejectingEngine):
            @RejectingEngine.weight_streaming_budget_v2.setter
            def weight_streaming_budget_v2(self, budget):
                if budget == 1000:
                    raise RuntimeError("allocation rejected")
                self._budget = budget

        saved = []
        with self.assertRaisesRegex(RuntimeError, "allocation rejected"):
            inspect_budget_rows(ThrowingEngine(), [0, 500, 1000], on_row=saved.append)
        self.assertEqual([row["actual_budget_bytes"] for row in saved], [0, 500])

    def test_rejects_invalid_budgets_and_nonstreaming_engine(self):
        for budgets in ([], [0, 0], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                inspect_budget_rows(Engine(), budgets)
        engine = Engine()
        engine.streamable_weights_size = 0
        with self.assertRaisesRegex(ValueError, "no streamable"):
            inspect_budget_rows(engine, [0])

    def test_rejects_scratch_omitted_from_context_query(self):
        class BrokenEngine(Engine):
            device_memory_size_v2 = 1
        with self.assertRaisesRegex(ValueError, "omitted"):
            inspect_budget_rows(BrokenEngine(), [0])

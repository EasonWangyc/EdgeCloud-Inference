"""Check paged ONNX graph bindings without requiring model weights or ONNX locally."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.experiments.prepare_edgellm_paged_onnx import (
    expected_attention_count, validate_paged_graph_contract,
)


def value(name: str, dimensions: list[int | str], dtype: int = 6) -> SimpleNamespace:
    dims = [SimpleNamespace(dim_value=item if isinstance(item, int) else 0,
                            dim_param=item if isinstance(item, str) else "")
            for item in dimensions]
    return SimpleNamespace(name=name, type=SimpleNamespace(
        tensor_type=SimpleNamespace(elem_type=dtype,
                                    shape=SimpleNamespace(dim=dims))))


def paged_model() -> SimpleNamespace:
    nodes = [SimpleNamespace(
        op_type="AttentionPlugin",
        input=[f"input_{index}_{position}" for position in range(7)] + ["kv_page_table"],
        attribute=[SimpleNamespace(name="enable_paged_kv", i=1)],
    ) for index in range(2)]
    cache_shape = [2, 32, 128, 8, 128]
    return SimpleNamespace(graph=SimpleNamespace(
        node=nodes,
        input=[value("kv_page_table", ["batch", 2, 8])]
        + [value(f"past_key_values_{index}", cache_shape) for index in range(2)],
        output=[value(f"present_key_values_{index}", cache_shape) for index in range(2)],
    ))


class PagedOnnxStageTests(unittest.TestCase):
    def test_model_layer_count_must_be_present_and_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"num_hidden_layers": 28}', encoding="utf-8")
            self.assertEqual(expected_attention_count(config), 28)
            for contents in ('{}', '{"num_hidden_layers": 0}',
                             '{"num_hidden_layers": true}'):
                config.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
                    expected_attention_count(config)

    def test_valid_paged_contract_records_binding_counts(self) -> None:
        result = validate_paged_graph_contract(
            paged_model(), attention_count=2, max_batch_size=4,
            max_pages_per_sequence=8,
            max_kv_pool_pages=32, int32_type=6,
        )
        self.assertEqual(result["kv_cache_input_count"], 2)
        self.assertEqual(result["kv_cache_output_count"], 2)

    def test_rejects_wrong_page_table_binding_and_kv_shape(self) -> None:
        for mutation, message in (
            (lambda model: model.graph.node[0].input.__setitem__(7, "wrong_table"), "page-table binding"),
            (lambda model: setattr(model.graph.input[0].type.tensor_type, "elem_type", 1), "dtype or shape"),
            (lambda model: setattr(model.graph.input[0].type.tensor_type.shape.dim[2], "dim_value", 7), "dtype or shape"),
            (lambda model: setattr(model.graph.input[0].type.tensor_type.shape.dim[0], "dim_param", ""), "dtype or shape"),
            (lambda model: setattr(model.graph.output[0].type.tensor_type.shape.dim[1], "dim_value", 16), "pool shape"),
            (lambda model: setattr(model.graph.output[0].type.tensor_type.shape.dim[4], "dim_value", 64), "shape or dtype differs"),
            (lambda model: setattr(model.graph.output[0], "name", "present_key_values_1"), "duplicate or unnamed"),
            (lambda model: setattr(model.graph.output[0], "name", "present_key_values_2"), "layers do not match"),
            (lambda model: model.graph.output.pop(), "bindings do not match"),
        ):
            with self.subTest(message=message):
                model = copy.deepcopy(paged_model())
                mutation(model)
                with self.assertRaisesRegex(ValueError, message):
                    validate_paged_graph_contract(
                        model, attention_count=2, max_batch_size=4,
                        max_pages_per_sequence=8,
                        max_kv_pool_pages=32, int32_type=6,
                    )

    def test_rejects_static_page_table_batch_below_build_limit(self) -> None:
        model = paged_model()
        batch_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
        batch_dimension.dim_param = ""
        batch_dimension.dim_value = 1
        with self.assertRaisesRegex(ValueError, "kv_page_table dtype or shape"):
            validate_paged_graph_contract(
                model, attention_count=2, max_batch_size=4,
                max_pages_per_sequence=8, max_kv_pool_pages=32, int32_type=6,
            )


if __name__ == "__main__":
    unittest.main()

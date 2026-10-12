from __future__ import annotations

from types import SimpleNamespace as NS
import unittest

from scripts.inspect_onnx_output_projection import projection_from_graph


def graph():
    return NS(
        output=[NS(name="logits")],
        node=[NS(name="projection", domain="", op_type="MatMul",
                 input=["hidden", "val_1246"], output=["linear"]),
              NS(name="cast", domain="", op_type="Cast",
                 input=["linear"], output=["logits"])],
        initializer=[NS(name="val_1246", dims=[2048, 151936], data_type=10,
                        data_location=1, raw_data=b"",
                        external_data=[NS(key="location", value="model.onnx.data")])],
    )


class OnnxOutputProjectionTests(unittest.TestCase):
    def test_anonymous_external_weight_is_found_via_output_path(self):
        result = projection_from_graph(graph())
        self.assertEqual(result["weight"]["name"], "val_1246")
        self.assertEqual(result["weight"]["dimensions"], [2048, 151936])
        self.assertEqual(result["weight"]["raw_data_bytes"], 0)
        self.assertEqual(result["weight"]["external_data"]["location"], "model.onnx.data")
        self.assertEqual([node["op_type"] for node in result["output_to_projection_path"]],
                         ["Cast", "MatMul"])

    def test_missing_or_transformed_weight_is_not_guessed(self):
        value = graph()
        value.initializer = []
        with self.assertRaisesRegex(ValueError, "direct initializer"):
            projection_from_graph(value)

    def test_ambiguous_or_cyclic_graph_is_rejected(self):
        value = graph()
        value.node.append(value.node[1])
        with self.assertRaisesRegex(ValueError, "duplicate producer"):
            projection_from_graph(value)
        value = graph()
        value.node[1].input = ["logits"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            projection_from_graph(value)

    def test_unsupported_operator_and_absent_output_are_rejected(self):
        value = graph()
        value.node[1].op_type = "Softmax"
        with self.assertRaisesRegex(ValueError, "unsupported output path"):
            projection_from_graph(value)
        with self.assertRaisesRegex(ValueError, "no output"):
            projection_from_graph(graph(), "absent")


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from scripts.quantize_qwen3_vl_int4_awq_domain import (
    _iter_texts,
    _quantization_kwargs,
    _validate_quantize_export_interface,
)


class DomainQuantizationDataTests(unittest.TestCase):
    def test_loads_non_empty_text_rows(self) -> None:
        self.assertEqual(
            list(_iter_texts([{"text": "parking prompt"}])),
            ["parking prompt"],
        )

    def test_rejects_missing_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires non-empty text"):
            list(_iter_texts([{"sample_id": "missing-text"}]))

    def test_formal_quantization_kwargs_keep_lm_head_unquantized(self) -> None:
        kwargs = _quantization_kwargs(
            model_dir=Path("model"),
            output_dir=Path("output"),
            text_dataset=object(),
            num_samples=16,
            lm_head_quantization=None,
        )
        self.assertNotIn("lm_head_quantization", kwargs)

    def test_lm_head_candidate_is_explicit(self) -> None:
        kwargs = _quantization_kwargs(
            model_dir=Path("model"),
            output_dir=Path("output"),
            text_dataset=object(),
            num_samples=16,
            lm_head_quantization="int4_awq",
        )
        self.assertEqual(kwargs["lm_head_quantization"], "int4_awq")

    def test_lm_head_candidate_requires_exporter_support(self) -> None:
        def supported(*, lm_head_quantization: str) -> None:
            del lm_head_quantization

        _validate_quantize_export_interface(
            supported,
            requested_lm_head_quantization="int4_awq",
        )

        def unsupported(*, quantization: str) -> None:
            del quantization

        with self.assertRaisesRegex(RuntimeError, "does not support"):
            _validate_quantize_export_interface(
                unsupported,
                requested_lm_head_quantization="int4_awq",
            )


if __name__ == "__main__":
    unittest.main()

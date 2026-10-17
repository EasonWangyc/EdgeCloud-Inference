from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.summarize_tensorrt_nsight_trace import classify_kernel, summarize_trace


class NsightTraceSummaryTests(unittest.TestCase):
    def test_edge_llm_decode_xqa_cubin_has_generic_kernel_name(self):
        self.assertEqual(classify_kernel("kernel_mha"), "attention_xqa")
        self.assertEqual(classify_kernel("other_kernel_mha_variant"), "other_kernel")

    def test_actual_fusion_gemvx_is_separate_from_int4_and_generic_gemv(self):
        name = "sm50_xmma_cublas_gemvx_f16f16_f32_f32_nn_n_int32_unit_n_launch_param64x2x1_strided_copyx_unit_stride_execute_kernel_trt"
        self.assertEqual(classify_kernel(name), "fp16_gemv")
        self.assertEqual(classify_kernel("cublas_gemv_fp16"), "other_kernel")

    def test_observed_level1_half_gemm_names_are_not_lost_in_other_kernel(self):
        names = (
            "trt_ampere_h16816gemm_128x64_ldg8_tn_v1",
            "trt_ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_gelu_tn_v1",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(classify_kernel(name), "fp16_gemm")
        self.assertEqual(classify_kernel("ampere_s16816gemm_128x64_nn"), "other_kernel")

    def test_summarizes_kernel_categories_and_excludes_transfers_from_kernel_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace = Path(temporary_directory) / "trace.csv"
            with trace.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Duration (ns)", "Name"])
                writer.writeheader()
                writer.writerow({"Duration (ns)": "1000000", "Name": "gemm_w4a16_T2"})
                writer.writerow({"Duration (ns)": "2000000", "Name": "fmha_v2_flash_attention"})
                writer.writerow({"Duration (ns)": "3000000", "Name": "topKStage1"})
                writer.writerow({"Duration (ns)": "4000000", "Name": "[CUDA memcpy Device-to-Host]"})

            result = summarize_trace(trace, label="control")

        self.assertEqual(result["input"]["label"], "control")
        self.assertEqual(result["trace"]["event_count"], 4)
        self.assertEqual(result["trace"]["kernel_duration_ms"], 6.0)
        self.assertEqual(result["trace"]["transfer_duration_ms"], 4.0)
        self.assertEqual(result["categories"]["int4_w4a16"]["count"], 1)
        self.assertEqual(result["categories"]["attention_fmha"]["total_ms"], 2.0)
        self.assertEqual(result["categories"]["sampling"]["total_ms"], 3.0)
        self.assertEqual(result["categories"]["memcpy"]["count"], 1)
        self.assertEqual(result["top_kernels"][0]["name"], "topKStage1")

    def test_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace = Path(temporary_directory) / "trace.csv"
            trace.write_text("Name\nfoo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                summarize_trace(trace)

    def test_actual_jetson_gemv_and_fp16_gemm_symbols_have_distinct_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            with trace.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Duration (ns)", "Name"])
                writer.writerow([110000000, "sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize256x64x32_stage3"])
                writer.writerow([70000, "void trt_edgellm::kernel::gemv_kernel<(int)2, (int)1, (int)256, (int)128>(const __half *, const unsigned int *, const __half *, __half *, int, int)"])
                writer.writerow([10000, "cublas_gemv_fp16"])
            result = summarize_trace(trace)
        self.assertEqual(result["categories"]["fp16_gemm"]["total_ms"], 110.0)
        self.assertEqual(result["categories"]["int4_w4a16"]["total_ms"], 0.07)
        self.assertEqual(result["categories"]["other_kernel"]["count"], 1)

    def test_nonfinite_durations_cannot_produce_a_valid_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            for value in ("nan", "inf", "-inf"):
                with self.subTest(value=value):
                    trace.write_text(f"Duration (ns),Name\n{value},kernel\n")
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        summarize_trace(trace)

    def test_cuda_api_waits_cannot_be_reported_as_gpu_kernel_work(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "api.csv"
            trace.write_text("Start (ns),Duration (ns),Name,Result,Pid,Tid\n0,100000000,cudaStreamSynchronize,0,1,1\n")
            with self.assertRaisesRegex(ValueError, "API trace"):
                summarize_trace(trace)

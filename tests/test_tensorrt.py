from __future__ import annotations

import json
import hashlib
import re
import struct
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.tensorrt import (
    TensorRTValidationError,
    build_engine_provenance,
    estimate_kv_cache_bytes,
    load_benchmark_samples,
    sha256_file,
    summarize_benchmark_samples,
    validate_profile_contract,
)
from scripts.evaluate_tensorrt_candidate import evaluate_candidate
from scripts.audit_int4_gemm_candidates import audit_candidates, estimate_shared_memory_bytes
from scripts.build_edgellm_vlm_engines import (
    _new_build_report,
    _validate_quantization_provenance,
    _validate_reduced_vocab_coverage_report,
    _validate_reduced_vocab_for_build,
    _validate_timing_cache_binding,
)
from scripts.build_edgellm_vlm_engines import build_commands
from scripts.compare_tensorrt_engine_inspectors import compare_inspectors
from scripts.compare_tensorrt_benchmarks import compare_benchmarks
from scripts.compare_tensorrt_study_reports import compare_study_reports
from scripts.benchmark_edgellm import main as benchmark_edgellm_main
from scripts.record_engine_provenance import main as record_engine_provenance_main
from scripts.run_edgellm_benchmark import benchmark, run_sample
from scripts.summarize_edgellm_runtime_log import (
    compare_summaries,
    summarize_log,
)
from scripts.validate_tensorrt_provenance import validate_provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TensorRTConfigAndEvidenceTests(unittest.TestCase):
    def test_reduced_vocab_coverage_gate_checks_map_and_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "coverage.json"
            payload = {
                "schema_version": "parksight_reduced_vocab_coverage_v1",
                "valid": True,
                "samples_with_missing_tokens": 0,
                "map": {"sha256": "a" * 64},
                "token_coverage_fraction": 1.0,
                "sample_coverage_fraction": 1.0,
                "reference_sample_count": 20,
                "total_reference_tokens": 632,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = _validate_reduced_vocab_coverage_report(
                path,
                source_map_sha256="a" * 64,
            )
            self.assertTrue(result["map_sha256_verified"])
            self.assertEqual(result["reference_sample_count"], 20)

            payload["valid"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid=true"):
                _validate_reduced_vocab_coverage_report(
                    path,
                    source_map_sha256="a" * 64,
                )

    def test_quantization_provenance_gate_checks_lm_head_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "calibration_provenance.json"
            payload = {
                "status": "succeeded",
                "edge_llm_revision": "edge-revision",
                "quantization": "int4_awq",
                "lm_head_precision": "int4_awq",
                "kv_cache_quantization": None,
                "calibration_rows": 16,
                "calibration_workload_identity": "workload-v1",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = _validate_quantization_provenance(
                path,
                expected_edge_llm_revision="edge-revision",
                expected_lm_head_precision="int4_awq",
            )
            self.assertEqual(report["lm_head_precision"], "int4_awq")
            self.assertEqual(report["calibration_rows"], 16)

            payload["lm_head_precision"] = "fp16"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lm_head precision mismatch"):
                _validate_quantization_provenance(
                    path,
                    expected_edge_llm_revision="edge-revision",
                    expected_lm_head_precision="int4_awq",
                )

    def test_reduced_vocab_build_gate_checks_source_and_export_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            onnx_llm = root / "onnx" / "llm"
            source.mkdir()
            onnx_llm.mkdir(parents=True)
            header = json.dumps(
                {
                    "vocab_map": {
                        "dtype": "I32",
                        "shape": [2],
                        "data_offsets": [0, 8],
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            map_bytes = (
                struct.pack("<Q", len(header))
                + header
                + struct.pack("<2i", 0, 1)
            )
            metadata = json.dumps(
                {"vocab_size": 16, "reduced_vocab_size": 2}
            )
            (source / "vocab_map.safetensors").write_bytes(map_bytes)
            (source / "reduced_vocab.json").write_text(metadata, encoding="utf-8")
            (source / "selection_report.json").write_text(
                json.dumps(
                    {
                        "map": {
                            "sha256": hashlib.sha256(map_bytes).hexdigest(),
                            "count": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (onnx_llm / "config.json").write_text(
                json.dumps({"reduced_vocab_size": 2}), encoding="utf-8"
            )
            (onnx_llm / "vocab_map.safetensors").write_bytes(map_bytes)
            (onnx_llm / "reduced_vocab.json").write_text(metadata, encoding="utf-8")

            report = _validate_reduced_vocab_for_build(
                source,
                onnx_llm,
                required_token_ids=(1,),
            )
            self.assertEqual(report["reduced_vocab_size"], 2)
            self.assertEqual(report["map"]["size_bytes"], len(map_bytes))
            self.assertEqual(report["required_token_ids"], [1])
            with self.assertRaisesRegex(ValueError, "multiple of 128"):
                _validate_reduced_vocab_for_build(
                    source,
                    onnx_llm,
                    lm_head_precision="int4_awq",
                )

            (onnx_llm / "vocab_map.safetensors").write_bytes(map_bytes + b"x")
            with self.assertRaisesRegex(ValueError, "does not match source map"):
                _validate_reduced_vocab_for_build(source, onnx_llm)

            (onnx_llm / "vocab_map.safetensors").write_bytes(map_bytes)
            (onnx_llm / "config.json").write_text(
                json.dumps({"reduced_vocab_size": 4}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config reduced_vocab_size"):
                _validate_reduced_vocab_for_build(
                    source,
                    onnx_llm,
                    required_token_ids=(1,),
                )

            (onnx_llm / "config.json").write_text(
                json.dumps({"reduced_vocab_size": 2}), encoding="utf-8"
            )
            (source / "selection_report.json").write_text(
                json.dumps({"map": {"sha256": "wrong", "count": 2}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "selection report does not match"):
                _validate_reduced_vocab_for_build(source, onnx_llm)

    def test_registered_binding_cache_patch_targets_post_pinned_context_api(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0017-cache-registered-bindings.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        # 0017 is documented as applying after 0014, which changes mContext
        # from unique_ptr to a raw IExecutionContext pointer.
        self.assertIn(
            "-    if (!mRegistry.bindAll(mContext, map, dims))",
            patch_text,
        )
        self.assertNotIn("mRegistry.bindAll(mContext.get(), map, dims)", patch_text)

    def test_redundant_scan_patch_refreshes_binding_cache_before_early_return(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0020-skip-redundant-registry-scan.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("if (mCacheBindingState)", patch_text)
        self.assertIn("refreshPreparedBindingState();", patch_text)
        self.assertLess(
            patch_text.index("refreshPreparedBindingState();"),
            patch_text.index("return true;"),
        )

    def test_safe_redundant_scan_patch_has_consistent_unified_diff_hunks(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0023-safe-skip-fallback-binding-scan.patch"
        )
        lines = patch_path.read_text(encoding="utf-8").splitlines()
        hunk_headers = [
            (index, line)
            for index, line in enumerate(lines)
            if line.startswith("@@ ")
        ]
        self.assertEqual(len(hunk_headers), 3)
        hunk_pattern = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
        )
        for hunk_index, (header_index, header) in enumerate(hunk_headers):
            match = hunk_pattern.match(header)
            self.assertIsNotNone(match, header)
            assert match is not None
            old_expected = int(match.group(2) or "1")
            new_expected = int(match.group(4) or "1")
            boundaries = [
                index
                for index in range(header_index + 1, len(lines))
                if lines[index].startswith(("@@ ", "diff --git "))
            ]
            end_index = boundaries[0] if boundaries else len(lines)
            body = lines[header_index + 1 : end_index]
            old_actual = sum(1 for line in body if line.startswith((" ", "-")))
            new_actual = sum(1 for line in body if line.startswith((" ", "+")))
            self.assertEqual(old_actual, old_expected, header)
            self.assertEqual(new_actual, new_expected, header)
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertIn("refreshPreparedBindingState();", patch_text)

    def test_greedy_argmax_patch_is_pinned_and_targets_sampling_only(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0024-greedy-argmax-fast-path.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "915b272a9efdf0e0e93f978304ef2c90e2c398fb66511887f1754d8d1fb0f5d3",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("greedyTop1Kernel", patch_text)
        self.assertIn("topK == 1 && !topKValues.has_value()", patch_text)
        self.assertNotIn("sampling.h", patch_text)

    def test_device_token_feedback_patch_is_pinned_and_keeps_debug_fallback(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0025-device-token-feedback.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "1c3371fce51a10983288f963683c8da08d2c63552cc6825836e1f1bda462cd3c",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("cudaMemcpyDeviceToDevice", patch_text)
        self.assertIn("context.layerDebugger != nullptr || activeBatchSize != 1", patch_text)
        self.assertIn("cudaMemcpyHostToDevice", patch_text)

    def test_dynamic_batch_token_feedback_compacts_sampling_indices(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0026-device-token-feedback-dynamic-batch.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "83dd3c885285d3dc636072d9cb2f4cac75bbe1bd469de4fe6fa6dbcb41f607fc",
        )
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("compactTensorBatch", patch_text)
        self.assertIn("onBatchEvict", patch_text)
        self.assertIn("+    if (context.layerDebugger != nullptr)\n", patch_text)
        self.assertNotIn("+    if (context.layerDebugger != nullptr || activeBatchSize != 1)", patch_text)

    def test_fused_greedy_reduced_vocab_map_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0027-fused-greedy-reduced-vocab-map.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "c0d0d1da7911bbf9b805aa3028f864191f2bf747e10470b2b66578a5de5efaa8",
        )
        self.assertEqual(patch_text.count("diff --git "), 4)
        self.assertIn("reducedVocabMapping", patch_text)
        self.assertIn("greedyTop1Kernel", patch_text)
        self.assertIn("topKtopPSamplingFromLogits", patch_text)
        self.assertIn("mapReducedVocabToFullVocab", patch_text)

    def test_xqa_selection_diagnostic_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0028-xqa-selection-diagnostic.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "6d4d075a2346cd05522fb7caaf9a0fb337ebd13aa577d8e37b8fc54a94a210b0",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_LOG_XQA_SELECTION", patch_text)
        self.assertIn("mFuncName", patch_text)
        self.assertIn("mSelectionReported", patch_text)
        self.assertIn("LOG_INFO(\"XQA selected function=", patch_text)

    def test_xqa_thread_local_list_cache_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0036-xqa-thread-local-list-cache.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "7e4f8f9c7003aa1c5e730e84d3d808515cc9496d989ff32d434dbedd858d5089",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("thread_local XQAKernelLoadHashKey", patch_text)
        self.assertIn("avoid taking the loader mutex", patch_text)

    def test_xqa_selection_cache_patch_is_pinned_and_keyed(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0037-xqa-selection-cache.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "a0510059ce895f4d330f41424167bf718a0a04f752e536b1bd0a9d646850dec0",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("XQAKernelSelectionCacheKey", patch_text)
        self.assertIn("thread_local XQAKernelFuncInfo cachedKernelInfo", patch_text)
        self.assertIn("findKernelFunction", patch_text)
        self.assertIn("mSmVersion", patch_text)

    def test_int4_gemv_nperblock_patch_is_pinned_and_m1_only(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0038-int4-gemv-nperblock4.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "c983a30fbe442d0485197e7b5151ebf908f5b94e71c55364f7896956458b953c",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_INT4_GEMV_N_PER_BLOCK", patch_text)
        self.assertIn("gemv_kernel<4, 1, BLOCK_SIZE, 128>", patch_text)
        self.assertIn("m == 1 && n % 16 == 0", patch_text)
        self.assertIn("no-tail indexing remains valid", patch_text)

    def test_int4_gemv_block_size_patch_is_pinned_and_default_off(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0049-int4-gemv-block-size-candidate-v2.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "77bde41d9f40c258d889b2841cb8186abeb1b71f4add4da341831d4d1c647f69",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_INT4_GEMV_BLOCK_SIZE", patch_text)
        self.assertIn("launch_gemv_for_batch<2, 128>", patch_text)
        self.assertIn("launch_gemv_for_batch<2, 512>", patch_text)
        self.assertIn("existing 256-thread launch as the default", patch_text)

    def test_int4_gemm_cta_n256_patch_is_pinned_and_build_time_only(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0039-int4-gemm-cta-n256.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "de4e13ed18e412b444af154cc6bba9f32d9ab35c1d06a750997eede748266f62",
        )
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("kGemmCtaN = 256", patch_text)
        self.assertIn("requires N divisible by kernel::kGemmCtaN", patch_text)
        self.assertIn("0039 candidate", patch_text)
        self.assertNotIn("EDGELLM_INT4_GEMM_CTA_N", patch_text)

    def test_int4_gemm_cta_n512_patch_is_pinned_and_build_time_only(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0045-int4-gemm-cta-n512.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "3d442c309060b231ffa699fdcea33926b73f68a7a66c75cd8968684b4577fe2e",
        )
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("kGemmCtaN = 512", patch_text)
        self.assertIn("0045 candidate", patch_text)
        self.assertNotIn("EDGELLM_INT4_GEMM_CTA_N", patch_text)
        # The v0.9.1 kernel keeps STAGES=4 and has a compile-time 99 KiB
        # shared-memory assertion. CTA_N=512 therefore cannot be compiled as
        # a standalone candidate; keep this as a negative-control patch rather
        # than placing it in the build queue.
        shared_memory_bytes = (64 * 64 + 512 * 64 // 4 + 512) * 4 * 2
        self.assertEqual(shared_memory_bytes, 102400)
        self.assertGreaterEqual(shared_memory_bytes, 99 * 1024)

    def test_qwen3_vl_int4_cta_n256_shape_audit(self) -> None:
        # Fixed Qwen3-VL-2B text-backbone INT4 linear output dimensions. The FP16
        # lm_head vocabulary dimension is intentionally not part of this plugin audit.
        int4_output_dimensions = (2048, 1024, 6144, 2048)
        self.assertTrue(all(dimension % 256 == 0 for dimension in int4_output_dimensions))

    def test_qwen3_vl_int4_cta_n512_shape_audit(self) -> None:
        int4_output_dimensions = (2048, 1024, 6144, 2048)
        self.assertTrue(all(dimension % 512 == 0 for dimension in int4_output_dimensions))

    def test_int4_gemm_candidate_resource_audit_filters_cta_n512(self) -> None:
        report = audit_candidates(output_dimensions=(2048, 1024, 6144))
        records = {record["candidate_id"]: record for record in report["candidates"]}
        self.assertTrue(records["cta_n256"]["build_queue_eligible"])
        self.assertTrue(records["cta_m128"]["build_queue_eligible"])
        self.assertTrue(records["cta_k128"]["build_queue_eligible"])
        self.assertFalse(records["cta_n512"]["build_queue_eligible"])
        self.assertEqual(records["cta_n512"]["shared_memory_bytes"], 102400)
        self.assertEqual(estimate_shared_memory_bytes(cta_m=64, cta_n=128, cta_k=128, stages=4), 99328)

    def test_int4_gemm_cta_m128_patch_is_pinned_and_independent(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0041-int4-gemm-cta-m128.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "c9a9680ea61df7b0ddba344f59c39ad538bfddd832e9dcdeeb2502de64d9a531",
        )
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("kGemmCtaM = 128", patch_text)
        self.assertIn("constexpr int CTA_M = kGemmCtaM", patch_text)
        self.assertIn("0041 candidate", patch_text)
        self.assertNotIn("kGemmCtaN = 256", patch_text)
        shared_memory_bytes = (128 * 64 + 128 * 64 // 4 + 128) * 4 * 2
        self.assertEqual(shared_memory_bytes, 82944)
        self.assertLess(shared_memory_bytes, 99 * 1024)

    def test_int4_gemm_cta_k128_patch_is_pinned_and_memory_bounded(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0042-int4-gemm-cta-k128.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "adc42690e2d99455bc3b23080749bf0bbb5b44d61cc298564b5bb42131e61b24",
        )
        self.assertEqual(patch_text.count("diff --git "), 2)
        self.assertIn("kGemmCtaK = 128", patch_text)
        self.assertIn("constexpr int CTA_K = kGemmCtaK", patch_text)
        self.assertIn("0042 candidate", patch_text)
        self.assertNotIn("kGemmCtaM = 128", patch_text)
        shared_memory_bytes = (64 * 128 + 128 * 128 // 4 + 128) * 4 * 2
        self.assertEqual(shared_memory_bytes, 99328)
        self.assertLess(shared_memory_bytes, 99 * 1024)

    def test_fmha_selection_cache_patch_is_pinned_and_keyed(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0040-fmha-selection-cache.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "4e369509ee1f485c624d1ff344d0d967d9c30290d3fdc10c97df2fa08a00a63b",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("thread_local FMHAKernelLoadHashKey", patch_text)
        self.assertIn("thread_local FMHAKernelHashKey", patch_text)
        self.assertIn("findKernelFunction", patch_text)
        self.assertIn("immutable hash key", patch_text)

    def test_fmha_head128_tiled_patch_is_pinned_and_targeted(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0043-fmha-head128-tiled-candidate.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "4f7cfb1603895a9410b06808389f26cee241874e24989050c6f489fcb2ec7f38",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_FMHA_FORCE_GRANULAR_TILING", patch_text)
        self.assertIn("mSmVersion == fmha_v2::kSM_87", patch_text)
        self.assertIn("mHeadSize == 128", patch_text)
        self.assertIn("mPaddedSequenceLen > 64", patch_text)
        self.assertIn("forced tiled head_dim=128 candidate", patch_text)

    def test_greedy_argmax_block_size_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0029-greedy-argmax-block-size.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "dfa1c4b38e4f29daec0cb209b921d2cc0b7634b805d1bcd9e54c9ec3eca2201f",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE", patch_text)
        self.assertIn("greedyTop1Kernel<1024>", patch_text)
        self.assertIn("greedyTop1Kernel<256>", patch_text)

    def test_greedy_warp_reduction_patch_is_pinned_and_default_off(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0035-greedy-warp-reduction.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "2f1a1f0c40eb2e89e2db689f017774b732ca63714450fc2dea2ab40ccf38a66f",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("greedyTop1WarpKernel", patch_text)
        self.assertIn("__shfl_down_sync", patch_text)
        self.assertIn("EDGELLM_GREEDY_ARGMAX_IMPL", patch_text)
        self.assertIn("The CUB path remains the default control", patch_text)

    def test_direct_device_token_embedding_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0030-direct-device-token-embedding.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "b290e79c74c5bc78596d2c4de47a777c0a2285b0079cedee02097f20ba3e373b",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_DIRECT_DEVICE_TOKEN_EMBED", patch_text)
        self.assertIn("tokenIdsForEmbedding", patch_text)
        self.assertIn("Sampling indices shape mismatch", patch_text)

    def test_skip_unused_device_token_reshape_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0031-skip-unused-device-token-reshape.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "0d76fa1041f5805572299c2470206d6ff50a66f5e4bb186dee7e9b8e2cca857f",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("useDirectDeviceTokenEmbedding", patch_text)
        self.assertIn("mRuntime.preprocess.idsInput.reshape", patch_text)
        self.assertIn("samplingShape", patch_text)

    def test_configurable_int4_gemm_stages_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0032-configurable-int4-gemm-stages.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "6f23399ec8b124a25c2afab88a6984a39b06b6242fe5b50fdadea304287ae872",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("EDGELLM_INT4_GEMM_STAGES", patch_text)
        self.assertIn("launchGemmForwardCudaNew<2>", patch_text)
        self.assertIn("launchGemmForwardCudaNew<3>", patch_text)
        self.assertIn("launchGemmForwardCudaNew<4>", patch_text)

    def test_skip_empty_batch_compaction_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0033-skip-empty-batch-compaction.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "039c2773490b0c23bb3f24084e5471c277d88073df4c4a2287a1a588b2da1c6a",
        )
        self.assertEqual(patch_text.count("diff --git "), 4)
        self.assertIn("if (newActiveBatch > 0)", patch_text)
        self.assertIn("no-op compaction kernel", patch_text)
        self.assertIn("strategy.onBatchEvict", patch_text)

    def test_skip_empty_batch_eviction_sync_patch_is_pinned(self) -> None:
        patch_path = (
            REPOSITORY_ROOT
            / "patches"
            / "tensorrt-edge-llm"
            / "0034-skip-empty-batch-eviction-sync.patch"
        )
        patch_text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(
            sha256_file(patch_path),
            "9692e49d9c68accde98dfe5d56bb3ca0959f744cc7718449fb84015681d6a339",
        )
        self.assertEqual(patch_text.count("diff --git "), 1)
        self.assertIn("No fixed v0.9.1 strategy consumes the device mapping", patch_text)
        self.assertIn("if (newActiveBatch > 0)", patch_text)
        self.assertIn("cudaStreamSynchronize(context.stream)", patch_text)

    def test_build_report_tracks_only_tuning_environment(self) -> None:
        from argparse import Namespace

        args = Namespace(
            component="llm",
            max_batch_size=1,
            max_input_len=768,
            max_kv_cache_capacity=1024,
            min_image_tokens=8,
            max_image_tokens=2048,
            max_image_tokens_per_image=2048,
            workspace_limit_mib=1024,
            builder_optimization_level=2,
            profiling_verbosity="detailed",
            enable_weight_streaming=False,
            timing_cache=Path("/cache/jetson.cache"),
        )
        report = _new_build_report(
            args=args,
            edge_root=Path("/edge"),
            onnx_root=Path("/onnx"),
            engine_root=Path("/engine"),
            llm_command=["llm_build"],
            visual_command=["visual_build"],
            environment={
                "EDGELLM_PLUGIN_PATH": "/edge/plugin.so",
                "EDGELLM_BUILDER_OPT_LEVEL": "2",
                "EDGELLM_TIMING_CACHE_PATH": "/cache/jetson.cache",
                "UNRELATED_SECRET": "must-not-be-recorded",
            },
            actual_revision="edge-revision",
            expected_outputs=[Path("/engine/llm/llm.engine")],
        )
        self.assertEqual(report["builder_environment"]["EDGELLM_BUILDER_OPT_LEVEL"], "2")
        self.assertNotIn("UNRELATED_SECRET", report["builder_environment"])
        self.assertEqual(report["configuration"]["workspace_limit_mib"], 1024)
        self.assertEqual(report["configuration"]["builder_optimization_level"], 2)
        self.assertEqual(report["configuration"]["profiling_verbosity"], "detailed")
        self.assertEqual(
            report["configuration"]["timing_cache"], str(Path("/cache/jetson.cache"))
        )
        self.assertEqual(
            report["timing_cache"],
            {
                "path": str(Path("/cache/jetson.cache")),
                "exists": False,
                "size_bytes": None,
                "sha256": None,
            },
        )
        self.assertEqual(
            report["builder_environment"]["EDGELLM_TIMING_CACHE_PATH"],
            "/cache/jetson.cache",
        )
        self.assertEqual(report["configuration"]["edge_llm_revision"], "edge-revision")
        self.assertEqual(report["outputs"][0]["sha256"], None)

    def test_timing_cache_binding_requires_device_and_builder_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "timing-cache.binding.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_timing_cache_binding_v1",
                        "device": {
                            "gpu_name": "Jetson Orin Nano",
                            "cuda_version": "12.6",
                            "tensorrt_version": "10.3.0",
                        },
                        "builder_config": {
                            "component": "llm",
                            "max_batch_size": 1,
                            "max_input_len": 768,
                            "max_kv_cache_capacity": 1024,
                            "workspace_limit_mib": 1024,
                            "builder_optimization_level": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = _validate_timing_cache_binding(
                path,
                expected_builder_config={
                    "component": "llm",
                    "max_batch_size": 1,
                    "max_input_len": 768,
                    "max_kv_cache_capacity": 1024,
                    "workspace_limit_mib": 1024,
                    "builder_optimization_level": 1,
                },
            )
            self.assertEqual(report["device"]["gpu_name"], "Jetson Orin Nano")
            self.assertEqual(report["builder_config"]["builder_optimization_level"], 1)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["builder_config"]["builder_optimization_level"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BuilderConfig mismatch"):
                _validate_timing_cache_binding(
                    path,
                    expected_builder_config={
                        "component": "llm",
                        "max_batch_size": 1,
                        "max_input_len": 768,
                        "max_kv_cache_capacity": 1024,
                        "workspace_limit_mib": 1024,
                        "builder_optimization_level": 1,
                    },
                )

    def test_build_commands_keep_llm_and_visual_profiles_explicit(self) -> None:
        llm, visual = build_commands(
            edge_root=Path("/edge"),
            onnx_root=Path("/onnx"),
            engine_root=Path("/engine"),
            max_batch_size=1,
            max_input_len=768,
            max_kv_cache_capacity=1024,
            min_image_tokens=8,
            max_image_tokens=2048,
            max_image_tokens_per_image=2048,
        )
        self.assertIn("--maxInputLen", llm)
        self.assertIn("768", llm)
        self.assertIn("--maxKVCacheCapacity", llm)
        self.assertNotIn("--maxKVCacheCapacity", visual)

    def test_visual_timing_cache_bindings_are_component_specific(self) -> None:
        for level in (2, 3):
            with self.subTest(level=level):
                path = (
                    REPOSITORY_ROOT
                    / "configs"
                    / "tensorrt"
                    / f"timing_cache_binding_jetson_orin_nano_visual_level{level}.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["builder_config"]["component"], "visual")
                self.assertEqual(
                    payload["builder_config"]["builder_optimization_level"], level
                )
                self.assertEqual(payload["builder_config"]["max_input_len"], 1024)
                self.assertEqual(payload["builder_config"]["max_kv_cache_capacity"], 2048)

    def test_loads_pinned_tuning_matrix_and_builder_variants(self) -> None:
        from parksight_vlm.tensorrt import TensorRTTuningConfig

        config = TensorRTTuningConfig.load(
            REPOSITORY_ROOT / "configs" / "tensorrt" / "jetson_orin_nano_int4_v1.json"
        )
        self.assertEqual(config.fixed.max_input_len, 768)
        self.assertEqual(config.fixed.max_kv_cache_capacity, 1024)
        self.assertEqual(config.builder_optimization_levels, (0, 1, 2, 3))
        self.assertEqual(config.builder_variants()[1]["variant_id"], "builder_opt_1_workspace_1024")

    def test_loads_independent_runtime_tuning_matrix(self) -> None:
        from parksight_vlm.tensorrt import TensorRTRuntimeTuningConfig

        config = TensorRTRuntimeTuningConfig.load(
            REPOSITORY_ROOT
            / "configs"
            / "tensorrt"
            / "jetson_orin_nano_int4_runtime_v1.json"
        )
        self.assertEqual(config.fixed["max_kv_cache_capacity"], 1024)
        self.assertEqual(config.variants[0].variant_id, "control")
        self.assertTrue(config.variants[1].pin_optimization_profiles)
        self.assertTrue(config.variants[-1].cache_registered_bindings)
        self.assertFalse(
            config.to_mapping()["variants"][0]["skip_fallback_binding_scan"]
        )

    def test_loads_int4_gemv_runtime_sweep_as_independent_variants(self) -> None:
        from parksight_vlm.tensorrt import TensorRTRuntimeTuningConfig

        config = TensorRTRuntimeTuningConfig.load(
            REPOSITORY_ROOT
            / "configs"
            / "tensorrt"
            / "jetson_orin_nano_int4_gemv_runtime_v1.json"
        )
        self.assertEqual(
            [variant.variant_id for variant in config.variants],
            [
                "control_n2_block256",
                "block128_n2",
                "block512_n2",
                "n4_block256",
            ],
        )
        self.assertEqual(config.variants[0].int4_gemv_n_per_block, 2)
        self.assertEqual(config.variants[0].int4_gemv_block_size, 256)
        self.assertEqual(config.variants[1].int4_gemv_block_size, 128)
        self.assertEqual(config.variants[2].int4_gemv_block_size, 512)
        self.assertEqual(config.variants[3].int4_gemv_n_per_block, 4)
        self.assertEqual(
            config.to_mapping()["variants"][3]["int4_gemv_block_size"],
            256,
        )

    def test_rejects_invalid_benchmark_status_and_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "benchmark.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "repetition": 1,
                        "status": "unknown",
                        "output_tokens": 3,
                        "timings_ms": {"decode_ms": 10},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TensorRTValidationError):
                load_benchmark_samples(path)

    def test_benchmark_summary_separates_failed_samples_and_computes_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "benchmark.jsonl"
            rows = [
                {
                    "sample_id": "sample-1",
                    "repetition": 1,
                    "status": "completed",
                    "output_tokens": 20,
                    "timings_ms": {"prefill_ms": 100, "ttft_ms": 150, "decode_ms": 400, "end_to_end_ms": 600},
                },
                {
                    "sample_id": "sample-2",
                    "repetition": 1,
                    "status": "failed",
                    "output_tokens": None,
                    "timings_ms": {},
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            samples = load_benchmark_samples(path)
            summary = summarize_benchmark_samples(samples)

        execution = summary["execution"]
        self.assertEqual(execution["sample_count"], 2)
        self.assertEqual(execution["completed_sample_count"], 1)
        self.assertEqual(execution["failed_sample_ids"], ["sample-2"])
        self.assertEqual(execution["decode_tokens_per_second"], 50.0)
        self.assertAlmostEqual(execution["end_to_end_tokens_per_second"], 33.3333333333)
        self.assertIsNone(execution["cold_start_ms"])
        self.assertEqual(execution["failed_categories"], {})
        self.assertEqual(summary["warmup"]["sample_count"], 0)
        self.assertEqual(execution["repetitions"]["1"]["sample_count"], 2)
        self.assertEqual(
            execution["repetitions"]["1"]["stage_latency_ms"]["decode_ms"]["p50"],
            400.0,
        )

    def test_benchmark_summary_uses_separate_warmup_for_cold_start(self) -> None:
        from parksight_vlm.tensorrt import BenchmarkSample

        samples = (
            BenchmarkSample(
                sample_id="sample-1",
                repetition=1,
                status="completed",
                output_tokens=2,
                timings_ms={"end_to_end_ms": 30.0},
            ),
        )
        warmup = (
            BenchmarkSample(
                sample_id="warmup-1",
                repetition=1,
                status="completed",
                output_tokens=2,
                timings_ms={"end_to_end_ms": 80.0},
            ),
        )
        summary = summarize_benchmark_samples(samples, warmup_samples=warmup)

        self.assertEqual(summary["execution"]["cold_start_ms"], 80.0)
        self.assertEqual(summary["warmup"]["first_completed_end_to_end_ms"], 80.0)

    def test_benchmark_rates_do_not_pair_fields_across_requests(self) -> None:
        from parksight_vlm.tensorrt import BenchmarkSample

        samples = (
            BenchmarkSample("count-only", 1, "completed", 100, {}),
            BenchmarkSample("time-only", 1, "completed", None, {"decode_ms": 10}),
        )
        execution = summarize_benchmark_samples(samples)["execution"]
        self.assertIsNone(execution["decode_tokens_per_second"])
        self.assertEqual(execution["decode_tokens_per_second_sample_count"], 0)
        self.assertIsNone(execution["repetitions"]["1"]["decode_tokens_per_second"])

    def test_benchmark_rates_use_only_matching_request_pairs(self) -> None:
        from parksight_vlm.tensorrt import BenchmarkSample

        samples = (
            BenchmarkSample("measured", 1, "completed", 10,
                            {"decode_ms": 20, "end_to_end_ms": 40}, decode_tokens=9),
            BenchmarkSample("missing-time", 1, "completed", 1000, {}),
            BenchmarkSample("missing-count", 1, "completed", None, {"decode_ms": 1000}),
            BenchmarkSample("failed", 1, "failed", 100, {"decode_ms": 1}, decode_tokens=99),
        )
        execution = summarize_benchmark_samples(samples)["execution"]
        self.assertEqual(execution["decode_tokens_per_second"], 500)
        self.assertEqual(execution["measured_decode_tokens_per_second"], 450)
        self.assertEqual(execution["end_to_end_tokens_per_second"], 250)
        self.assertEqual(execution["measured_decode_tokens_per_second_sample_count"], 1)
        self.assertEqual(execution["repetitions"]["1"]["measured_decode_tokens_per_second"], 450)

    def test_benchmark_decode_count_is_optional_and_validated(self) -> None:
        from parksight_vlm.tensorrt import BenchmarkSample

        row = {"sample_id": "single", "repetition": 1, "status": "completed",
               "output_tokens": 1, "timings_ms": {"decode_ms": 0}}
        legacy = BenchmarkSample.from_mapping(row, 1)
        self.assertIsNone(legacy.decode_tokens)
        self.assertIsNone(summarize_benchmark_samples([legacy])["execution"]["measured_decode_tokens_per_second"])
        first_only = BenchmarkSample.from_mapping({**row, "decode_tokens": 0}, 1)
        self.assertIsNone(summarize_benchmark_samples([first_only])["execution"]["measured_decode_tokens_per_second"])
        for invalid in (True, -1, 1.5, 2):
            with self.subTest(count=invalid), self.assertRaises(TensorRTValidationError):
                BenchmarkSample.from_mapping({**row, "decode_tokens": invalid}, 1)

    def test_benchmark_summary_preserves_system_throughput_metadata(self) -> None:
        from parksight_vlm.tensorrt import BenchmarkSample

        samples = (
            BenchmarkSample(
                sample_id="sample-1",
                repetition=1,
                status="completed",
                output_tokens=4,
                timings_ms={"end_to_end_ms": 100.0},
            ),
        )
        summary = summarize_benchmark_samples(
            samples,
            metadata={
                "concurrency": 4,
                "steady_state_wall_clock_ms": 200.0,
                "steady_state_requests": 1,
                "steady_state_completed_requests": 1,
                "steady_state_output_tokens": 4,
                "aggregate_output_tokens_per_second": 20.0,
            },
        )

        execution = summary["execution"]
        self.assertEqual(execution["concurrency"], 4)
        self.assertEqual(execution["steady_state_requests"], 1)
        self.assertEqual(execution["steady_state_output_tokens"], 4)
        self.assertEqual(execution["aggregate_output_tokens_per_second"], 20.0)

    def test_benchmark_summary_can_attach_tegrastats_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "benchmark.jsonl"
            telemetry_path = root / "tegrastats.log"
            output_path = root / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "repetition": 1,
                        "status": "completed",
                        "output_tokens": 2,
                        "timings_ms": {"decode_ms": 20, "end_to_end_ms": 30},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            telemetry_path.write_text(
                "RAM 1000/8000MB SWAP 10/1000MB GR3D_FREQ 90% gpu@55.0C "
                "VDD_IN 12000mW/13000mW\n",
                encoding="utf-8",
            )
            self.assertEqual(
                benchmark_edgellm_main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--tegrastats",
                        str(telemetry_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["jetson_telemetry"]["sample_count"], 1)
        self.assertEqual(summary["jetson_telemetry"]["gpu_utilization_percent"]["mean"], 90.0)

    def test_benchmark_summary_embeds_engine_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "benchmark.jsonl"
            provenance_path = root / "provenance.json"
            output_path = root / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "repetition": 1,
                        "status": "completed",
                        "output_tokens": 2,
                        "timings_ms": {"decode_ms": 20, "end_to_end_ms": 30},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provenance_path.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_engine_provenance_v1",
                        "engine": {"sha256": "a" * 64, "size_bytes": 12},
                        "configuration": {"builder_optimization_level": 1},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                benchmark_edgellm_main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--provenance-json",
                        str(provenance_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            summary["metadata"]["engine_provenance"]["engine"]["sha256"], "a" * 64
        )
        self.assertEqual(
            summary["evidence_sources"]["provenance_json"], str(provenance_path)
        )

    def test_benchmark_summary_embeds_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "benchmark.jsonl"
            runtime_path = root / "runtime.json"
            output_path = root / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-1",
                        "repetition": 1,
                        "status": "completed",
                        "output_tokens": 2,
                        "timings_ms": {"decode_ms": 20, "end_to_end_ms": 30},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime_path.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_runtime_options_v1",
                        "engines": {
                            "llm": {"sha256": "a" * 64},
                            "visual": {"sha256": "b" * 64},
                        },
                        "options": {"pin_optimization_profiles": "enabled"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                benchmark_edgellm_main(
                    [
                        "--input-jsonl",
                        str(input_path),
                        "--runtime-metadata-json",
                        str(runtime_path),
                        "--output",
                        str(output_path),
                    ]
                ),
                0,
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            summary["metadata"]["runtime_metadata"]["options"][
                "pin_optimization_profiles"
            ],
            "enabled",
        )
        self.assertEqual(
            summary["evidence_sources"]["runtime_metadata_json"], str(runtime_path)
        )

    def test_low_level_runner_keeps_http_ttft_and_decode_separate(self) -> None:
        from parksight_vlm.inference.runtime import RuntimeGeneration, StageTimings

        class FakeBackend:
            def generate(self, *, image_path: Path, workload: object) -> RuntimeGeneration:
                return RuntimeGeneration(
                    raw_output="{}",
                    stage_timings=StageTimings(
                        time_to_first_token_ms=12.0,
                        decode_ms=20.0,
                        http_round_trip_ms=25.0,
                    ),
                    output_tokens=4,
                    decode_tokens=3,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = root / "image.jpg"
            image.write_bytes(b"test")
            parking_case = ParkingCase(
                case_id="case-1",
                image_ref=PurePosixPath("image.jpg"),
                source_group_id="group-1",
                split=DatasetSplit.TEST,
                reference_assessment=None,
            )
            row = run_sample(
                backend=FakeBackend(),
                case=parking_case,
                workload=object(),
                data_root=root,
                repetition=1,
            )

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["decode_tokens"], 3)
        self.assertEqual(row["timings_ms"]["decode_ms"], 20.0)
        self.assertEqual(row["timings_ms"]["http_round_trip_ms"], 25.0)
        self.assertEqual(row["timings_ms"]["time_to_first_token_ms"], 12.0)
        self.assertGreater(row["timings_ms"]["end_to_end_ms"], 0.0)

    def test_low_level_runner_persists_failed_warmup_before_aborting(self) -> None:
        class FailingBackend:
            def generate(self, *, image_path: Path, workload: object) -> object:
                raise RuntimeError("warmup backend unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "image.jpg").write_bytes(b"test")
            parking_case = ParkingCase(
                case_id="case-1",
                image_ref=PurePosixPath("image.jpg"),
                source_group_id="group-1",
                split=DatasetSplit.TEST,
                reference_assessment=None,
            )
            output = root / "benchmark.jsonl"
            warmup_output = root / "warmup.jsonl"
            with self.assertRaisesRegex(RuntimeError, "warm-up failed"):
                benchmark(
                    cases=[parking_case],
                    workload=object(),
                    backend=FailingBackend(),
                    data_root=root,
                    repetitions=1,
                    warmup=1,
                    output_jsonl=output,
                    warmup_output_jsonl=warmup_output,
                )
            rows = [
                json.loads(line)
                for line in warmup_output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertFalse(output.exists())

    def test_low_level_runner_supports_ordered_concurrent_dynamic_batch_probe(self) -> None:
        from parksight_vlm.inference.runtime import RuntimeGeneration, StageTimings

        created = []

        class FakeBackend:
            def __init__(self) -> None:
                created.append(self)

            def generate(self, *, image_path: Path, workload: object) -> RuntimeGeneration:
                return RuntimeGeneration(
                    raw_output="{}",
                    stage_timings=StageTimings(
                        time_to_first_token_ms=2.0,
                        decode_ms=4.0,
                    ),
                    output_tokens=2,
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in ("one.jpg", "two.jpg"):
                (root / name).write_bytes(b"test")
            cases = [
                ParkingCase(
                    case_id=case_id,
                    image_ref=PurePosixPath(image_name),
                    source_group_id=f"group-{case_id}",
                    split=DatasetSplit.TEST,
                    reference_assessment=None,
                )
                for case_id, image_name in (("one", "one.jpg"), ("two", "two.jpg"))
            ]
            output = root / "concurrent.jsonl"
            run_stats = benchmark(
                cases=cases,
                workload=object(),
                backend=FakeBackend(),
                data_root=root,
                repetitions=2,
                warmup=0,
                output_jsonl=output,
                warmup_output_jsonl=None,
                concurrency=2,
                backend_factory=FakeBackend,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["sample_id"] for row in rows], ["one", "two", "one", "two"])
        self.assertEqual([row["repetition"] for row in rows], [1, 1, 2, 2])
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        self.assertGreaterEqual(len(created), 1)
        self.assertEqual(run_stats["steady_state_requests"], 4)
        self.assertEqual(run_stats["steady_state_completed_requests"], 4)
        self.assertEqual(run_stats["steady_state_output_tokens"], 8)
        self.assertGreater(run_stats["aggregate_output_tokens_per_second"], 0.0)

    def test_low_level_runner_requires_backend_factory_for_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend_factory"):
            benchmark(
                cases=[],
                workload=object(),
                backend=object(),
                data_root=Path("."),
                repetitions=1,
                warmup=0,
                output_jsonl=Path("benchmark.jsonl"),
                warmup_output_jsonl=None,
                concurrency=2,
            )

    def test_engine_provenance_contains_hash_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = Path(temporary_directory) / "llm.engine"
            engine.write_bytes(b"engine")
            provenance = build_engine_provenance(
                engine_path=engine,
                metadata={"builder_optimization_level": 1, "cuda_graph": "enabled"},
            )
            self.assertEqual(provenance["engine"]["sha256"], sha256_file(engine))
            self.assertEqual(provenance["configuration"]["builder_optimization_level"], 1)

    def test_provenance_validator_rejects_fixed_profile_mismatch(self) -> None:
        from parksight_vlm.tensorrt import TensorRTTuningConfig

        tuning_config = TensorRTTuningConfig.load(
            REPOSITORY_ROOT / "configs" / "tensorrt" / "jetson_orin_nano_int4_v1.json"
        )
        provenance = {
            "schema_version": "parksight_tensorrt_engine_provenance_v1",
            "engine": {"sha256": "a" * 64},
            "configuration": {
                **tuning_config.fixed_mapping(),
                "max_kv_cache_capacity": 768,
            },
        }
        result = validate_provenance(provenance, tuning_config)
        self.assertFalse(result["valid"])
        self.assertIn("max_kv_cache_capacity", result["fixed_fields"]["mismatches"])

    def test_provenance_validator_checks_engine_file_hash(self) -> None:
        from parksight_vlm.tensorrt import TensorRTTuningConfig

        tuning_config = TensorRTTuningConfig.load(
            REPOSITORY_ROOT / "configs" / "tensorrt" / "jetson_orin_nano_int4_v1.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = Path(temporary_directory) / "llm.engine"
            engine.write_bytes(b"actual-engine")
            provenance = {
                "schema_version": "parksight_tensorrt_engine_provenance_v1",
                "engine": {"sha256": "a" * 64, "size_bytes": 999},
                "configuration": tuning_config.fixed_mapping(),
            }
            result = validate_provenance(
                provenance, tuning_config, engine_path=engine
            )

        self.assertFalse(result["valid"])
        self.assertFalse(result["engine_file"]["sha256_match"])
        self.assertFalse(result["engine_file"]["size_match"])
        self.assertIn(
            "recorded engine SHA-256 does not match engine file", result["reasons"]
        )

    def test_provenance_validator_can_use_visual_build_report(self) -> None:
        from scripts.validate_tensorrt_provenance import main as validate_main

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "visual.engine"
            provenance = root / "visual.provenance.json"
            build_report = root / "visual.build.json"
            validation = root / "visual.validation.json"
            engine.write_bytes(b"visual-engine")
            configuration = {
                "component": "visual",
                "builder_optimization_level": 2,
                "workspace_limit_mib": 1024,
                "max_batch_size": 1,
                "max_input_len": 1024,
                "max_kv_cache_capacity": 2048,
                "min_image_tokens": 8,
                "max_image_tokens": 2048,
                "max_image_tokens_per_image": 2048,
            }
            build_report.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_engine_build_v1",
                        "status": "succeeded",
                        "configuration": configuration,
                        "outputs": [
                            {
                                "path": str(engine.resolve()),
                                "size_bytes": engine.stat().st_size,
                                "sha256": sha256_file(engine),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record_engine_provenance_main(
                [
                    "--engine",
                    str(engine),
                    "--build-report",
                    str(build_report),
                    "--output",
                    str(provenance),
                ]
            )

            self.assertEqual(
                validate_main(
                    [
                        "--provenance",
                        str(provenance),
                        "--build-report",
                        str(build_report),
                        "--engine",
                        str(engine),
                        "--output",
                        str(validation),
                    ]
                ),
                0,
            )
            result = json.loads(validation.read_text(encoding="utf-8"))

        self.assertTrue(result["valid"])
        self.assertTrue(result["build_report"]["valid"])
        self.assertTrue(result["build_report"]["engine_output"]["sha256_match"])
        self.assertEqual(result["fixed_fields"]["expected"]["component"], "visual")

    def test_provenance_validator_rejects_visual_build_report_hash_mismatch(self) -> None:
        from scripts.validate_tensorrt_provenance import main as validate_main

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "visual.engine"
            provenance = root / "visual.provenance.json"
            build_report = root / "visual.build.json"
            validation = root / "visual.validation.json"
            engine.write_bytes(b"visual-engine")
            configuration = {
                "component": "visual",
                "builder_optimization_level": 3,
                "workspace_limit_mib": 1024,
                "max_batch_size": 1,
                "max_input_len": 1024,
                "max_kv_cache_capacity": 2048,
                "min_image_tokens": 8,
                "max_image_tokens": 2048,
                "max_image_tokens_per_image": 2048,
            }
            build_report.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_engine_build_v1",
                        "status": "succeeded",
                        "configuration": configuration,
                        "outputs": [
                            {
                                "path": str(engine.resolve()),
                                "size_bytes": engine.stat().st_size,
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            provenance.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_engine_provenance_v1",
                        "engine": {
                            "sha256": sha256_file(engine),
                            "size_bytes": engine.stat().st_size,
                        },
                        "configuration": configuration,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                validate_main(
                    [
                        "--provenance",
                        str(provenance),
                        "--build-report",
                        str(build_report),
                        "--engine",
                        str(engine),
                        "--output",
                        str(validation),
                    ]
                ),
                2,
            )
            result = json.loads(validation.read_text(encoding="utf-8"))

        self.assertFalse(result["valid"])
        self.assertFalse(result["build_report"]["valid"])
        self.assertIn(
            "build report engine SHA-256 does not match engine file",
            result["reasons"],
        )

    def test_provenance_validator_checks_reduced_vocabulary_hashes(self) -> None:
        from parksight_vlm.tensorrt import TensorRTTuningConfig

        tuning_config = TensorRTTuningConfig.load(
            REPOSITORY_ROOT / "configs" / "tensorrt" / "jetson_orin_nano_int4_v1.json"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "llm.engine"
            reduced_vocab = root / "reduced_vocab"
            reduced_vocab.mkdir()
            engine.write_bytes(b"engine")
            (reduced_vocab / "vocab_map.safetensors").write_bytes(b"map")
            (reduced_vocab / "reduced_vocab.json").write_text(
                json.dumps({"vocab_size": 151936, "reduced_vocab_size": 65536}),
                encoding="utf-8",
            )
            (reduced_vocab / "selection_report.json").write_text(
                json.dumps(
                    {
                        "map": {
                            "sha256": hashlib.sha256(b"map").hexdigest(),
                            "count": 65536,
                        }
                    }
                ),
                encoding="utf-8",
            )
            from scripts.record_engine_provenance import main as record_main

            record_main(
                [
                    "--engine",
                    str(engine),
                    "--output",
                    str(root / "provenance.json"),
                    "--reduced-vocab-dir",
                    str(reduced_vocab),
                ]
            )
            provenance = json.loads(
                (root / "provenance.json").read_text(encoding="utf-8")
            )
            result = validate_provenance(
                provenance,
                tuning_config,
                reduced_vocab_dir=reduced_vocab,
            )

        self.assertTrue(result["reduced_vocabulary_file"]["map"]["sha256_match"])
        self.assertTrue(result["reduced_vocabulary_file"]["metadata"]["size_match"])
        self.assertNotIn(
            "reduced vocabulary map SHA-256 does not match file", result["reasons"]
        )

    def test_record_engine_provenance_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "llm.engine"
            output = root / "provenance.json"
            engine.write_bytes(b"engine")
            self.assertEqual(
                record_engine_provenance_main(
                    ["--engine", str(engine), "--output", str(output)]
                ),
                0,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["engine"]["filename"], "llm.engine")

    def test_record_engine_provenance_can_bind_successful_build_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "visual.engine"
            output = root / "provenance.json"
            build_report = root / "build.json"
            engine.write_bytes(b"visual-engine")
            build_report.write_text(
                json.dumps(
                    {
                        "schema_version": "parksight_tensorrt_engine_build_v1",
                        "status": "succeeded",
                        "configuration": {
                            "component": "visual",
                            "builder_optimization_level": 2,
                            "workspace_limit_mib": 1024,
                            "min_image_tokens": 8,
                            "max_image_tokens": 2048,
                            "max_image_tokens_per_image": 2048,
                        },
                        "builder_environment": {
                            "EDGELLM_BUILDER_OPT_LEVEL": "2"
                        },
                        "outputs": [
                            {
                                "path": str(engine.resolve()),
                                "size_bytes": engine.stat().st_size,
                                "sha256": sha256_file(engine),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                record_engine_provenance_main(
                    [
                        "--engine",
                        str(engine),
                        "--output",
                        str(output),
                        "--build-report",
                        str(build_report),
                    ]
                ),
                0,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["configuration"]["component"], "visual")
        self.assertEqual(payload["configuration"]["max_image_tokens"], 2048)
        self.assertEqual(payload["build_report"]["component"], "visual")
        self.assertTrue(payload["build_report"]["engine_output_sha256_verified"])
        self.assertEqual(
            payload["evidence_sources"]["build_report"], str(build_report)
        )

    def test_record_engine_provenance_binds_reduced_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "llm.engine"
            reduced_vocab = root / "reduced_vocab"
            output = root / "provenance.json"
            engine.write_bytes(b"engine")
            reduced_vocab.mkdir()
            (reduced_vocab / "vocab_map.safetensors").write_bytes(b"map")
            (reduced_vocab / "reduced_vocab.json").write_text(
                json.dumps({"vocab_size": 151936, "reduced_vocab_size": 65536}),
                encoding="utf-8",
            )
            (reduced_vocab / "selection_report.json").write_text(
                json.dumps(
                    {
                        "map": {
                            "sha256": hashlib.sha256(b"map").hexdigest(),
                            "count": 65536,
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                record_engine_provenance_main(
                    [
                        "--engine",
                        str(engine),
                        "--output",
                        str(output),
                        "--reduced-vocab-dir",
                        str(reduced_vocab),
                    ]
                ),
                0,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["reduced_vocabulary"]["metadata"]["reduced_vocab_size"],
            65536,
        )
        self.assertTrue(
            payload["reduced_vocabulary"]["selection_report"][
                "map_sha256_verified"
            ]
        )
        self.assertEqual(
            payload["evidence_sources"]["reduced_vocab_dir"], str(reduced_vocab)
        )

    def test_candidate_gate_accepts_repeated_non_regression(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(result["reasons"], [])

    def test_candidate_gate_requires_active_paged_kv_when_requested(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            require_paged_kv_active=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn(
            "candidate runtime log summary is required to confirm paged KV",
            result["reasons"],
        )

        runtime_summary = {
            "attention": {"use_paged_kv_cache": False},
            "cuda_graph": {
                "decoder_capture_success": True,
                "replay_observed": True,
            },
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=runtime_summary,
            require_paged_kv_active=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn("candidate paged KV cache is not confirmed active", result["reasons"])

        active_without_pool_size = {
            "attention": {"use_paged_kv_cache": True},
            "cuda_graph": {
                "decoder_capture_success": True,
                "replay_observed": True,
            },
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=active_without_pool_size,
            require_paged_kv_active=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn("candidate paged KV pool page count is not confirmed", result["reasons"])

        active_with_pool_size = {
            "attention": {
                "use_paged_kv_cache": True,
                "max_kv_pool_pages": 16,
            },
            "cuda_graph": {
                "decoder_capture_success": True,
                "replay_observed": True,
            },
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=active_with_pool_size,
            require_paged_kv_active=True,
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(result["comparison"]["paged_kv_pool_pages"], 16)

    def test_candidate_gate_requires_runtime_and_log_evidence_for_fmha_tiling(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        runtime_summary = {
            "fmha": {"forced_tiled_head128_observed": True},
            "cuda_graph": {
                "decoder_capture_success": True,
                "replay_observed": True,
            },
        }

        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=runtime_summary,
            require_fmha_tiled_active=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn(
            "candidate runtime metadata is required to confirm FMHA variant",
            result["reasons"],
        )

        metadata = {
            "schema_version": "parksight_tensorrt_runtime_options_v1",
            "options": {"fmha_force_granular_tiling": "1"},
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=runtime_summary,
            candidate_runtime_metadata=metadata,
            require_fmha_tiled_active=True,
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertTrue(result["comparison"]["fmha_tiled_active"])

        runtime_summary["fmha"]["forced_tiled_head128_observed"] = False
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary=runtime_summary,
            candidate_runtime_metadata=metadata,
            require_fmha_tiled_active=True,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn(
            "candidate FMHA tiled selection marker is not observed",
            result["reasons"],
        )

    def test_candidate_gate_reports_available_stage_comparisons(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        for report, prefill, ttft, decode in (
            (baseline, 500, 550, 400),
            (candidate, 450, 490, 300),
        ):
            report["performance_metrics"]["stage_latency_ms"].update(
                {
                    "prefill_ms": {"count": 60, "p50": prefill, "p90": prefill + 20, "p99": prefill + 40},
                    "time_to_first_token_ms": {"count": 60, "p50": ttft, "p90": ttft + 20, "p99": ttft + 40},
                    "decode_ms": {"count": 60, "p50": decode, "p90": decode + 20, "p99": decode + 40},
                }
            )
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
        )
        stages = result["comparison"]["stage_latency_ms"]
        self.assertAlmostEqual(stages["prefill_ms"]["p50"]["improvement"], 0.1)
        self.assertAlmostEqual(
            stages["time_to_first_token_ms"]["p50"]["improvement"], 60 / 550
        )
        self.assertAlmostEqual(stages["decode_ms"]["p50"]["speedup"], 4 / 3)

    def test_candidate_gate_rejects_json_or_quality_regression(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=700,
            p90=800,
            quality=0.20,
            records=60,
            json_validity=0.95,
        )
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertTrue(any("JSON validity" in reason for reason in result["reasons"]))
        self.assertTrue(any("quality regression" in reason for reason in result["reasons"]))

    def test_candidate_gate_honors_optional_quality_and_completion_flags(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=700,
            p90=800,
            quality=0.20,
            records=60,
            json_validity=0.95,
        )
        candidate["performance_metrics"]["backend_completed_sample_count"] = 59
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            require_backend_completion=False,
            require_json_validity=False,
            require_quality_non_regression=False,
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["comparison"]["backend_completion_rate"], 59 / 60)
        self.assertEqual(result["policy"]["require_json_validity"], False)

    def test_candidate_gate_rejects_unconfirmed_decoder_graph(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            candidate_runtime_log_summary={
                "cuda_graph": {"decoder_capture_success": False}
            },
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertIn("decoder CUDA Graph capture", result["reasons"][0])
        self.assertIn("decoder CUDA Graph replay", result["reasons"][1])
        self.assertFalse(result["comparison"]["decoder_graph_replay_ok"])

    def test_candidate_gate_rejects_runtime_identity_mismatch(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        baseline["study_identity"]["runtime_identity"] = {
            "backend": "tensorrt_edge_llm",
            "backend_revision": "edge-a",
            "model_id": "qwen",
            "model_revision": "model-a",
            "adapter_revision": "adapter-a",
            "precision": "int4_awq",
        }
        candidate["study_identity"]["runtime_identity"] = {
            **baseline["study_identity"]["runtime_identity"],
            "model_revision": "model-b",
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertTrue(any("runtime identity mismatch" in reason for reason in result["reasons"]))

    def test_candidate_gate_rejects_failed_low_level_repetition_gate(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1000,
            quality=0.30,
            records=60,
        )
        comparison = {
            "execution": {
                "repetition_gate": {
                    "status": "fail",
                    "eligible": False,
                }
            }
        }
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak={"failure_summary": {}},
            benchmark_comparison=comparison,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["comparison"]["benchmark_repetition_gate_ok"])
        self.assertTrue(
            any("low-level benchmark repetition gate" in reason for reason in result["reasons"])
        )

    def test_candidate_gate_requires_soak_report(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        result = evaluate_candidate(baseline=baseline, candidate=candidate)
        self.assertFalse(result["promotion_eligible"])
        self.assertIn("soak report is required for promotion", result["reasons"])

    def test_candidate_gate_binds_soak_to_candidate_identity_and_sample_count(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        soak = _soak_report()
        soak["study_identity"]["power_mode"] = "30W_MODE_0"
        soak["records"] = [{} for _ in range(99)]
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=soak,
        )
        self.assertFalse(result["promotion_eligible"])
        self.assertTrue(any("soak runtime/workload identity mismatch" in reason for reason in result["reasons"]))
        self.assertTrue(any("at least 100 records" in reason for reason in result["reasons"]))

    def test_candidate_gate_accepts_explicit_matching_jetson_resource_summaries(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        baseline["performance_metrics"]["peak_memory_mb"] = None
        candidate["performance_metrics"]["peak_memory_mb"] = None
        baseline_summary = _jetson_runtime_summary(baseline, memory=4000.0)
        candidate_summary = _jetson_runtime_summary(candidate, memory=4050.0)
        result = evaluate_candidate(
            baseline=baseline,
            candidate=candidate,
            soak=_soak_report(),
            baseline_jetson_summary=baseline_summary,
            candidate_jetson_summary=candidate_summary,
        )
        self.assertTrue(result["promotion_eligible"])
        self.assertAlmostEqual(result["comparison"]["memory_regression"], 0.0125)
        self.assertEqual(
            result["comparison"]["resource_evidence"]["candidate"]["peak_memory_mb"],
            4050.0,
        )

    def test_candidate_gate_rejects_jetson_resource_record_count_mismatch(self) -> None:
        baseline = _study_report(
            study_id="baseline",
            repetitions=3,
            p50=1000,
            p90=1200,
            quality=0.30,
            records=60,
        )
        candidate = _study_report(
            study_id="candidate",
            repetitions=3,
            p50=800,
            p90=1100,
            quality=0.30,
            records=60,
        )
        with self.assertRaisesRegex(TensorRTValidationError, "record_count"):
            evaluate_candidate(
                baseline=baseline,
                candidate=candidate,
                soak=_soak_report(),
                baseline_jetson_summary=_jetson_runtime_summary(baseline, memory=4000.0),
                candidate_jetson_summary=_jetson_runtime_summary(candidate, memory=4050.0, records=59),
            )

    def test_runtime_log_summary_separates_fmha_and_confirms_graph_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "runtime.log"
            log.write_text(
                "\n".join(
                    [
                        "[INFO] LLMEngineConfig{ hiddenSize=2048 numKVHeads=8 headDim=128 maxInputLen=768 "
                        "kvCacheDtype=kHALF usePagedKVCache=false maxKVPoolPages=16 tokensPerPage=64 specDecodeType=0 }",
                        "[INFO] Loading FMHA cubin: index=8 sm=87 size=140832 function=fmha_head128_sm87",
                        "[WARNING] Skipping FMHA cubin rejected by the CUDA driver: index=18 sm=87 size=61472 function=fmha_mask_head256_sm87",
                        "[INFO] Number of aux streams is 1",
                        "[INFO] Number of total worker streams is 2",
                        "[INFO] CUDA graph enabled",
                        "[INFO] CUDA graph captured successfully.",
                        "[INFO] Successfully captured decoding CUDA graphs for active decoding strategies.",
                        "[INFO] profile switch call profile=0 elapsed_ms=0.321 success=true; this is host API-call time, not GPU completion time",
                        "[INFO] pinned profile initialization call profile=1 elapsed_ms=0.456 success=true; this is host API-call time, not GPU completion time",
                        "[12:31:16.787] [INFO] [TensorRT] Switching optimization profile from: 1 to 0",
                        "[12:31:17.447] [INFO] [TensorRT] Switching optimization profile from: 0 to 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_log(log)

        self.assertEqual(summary["fmha"]["loaded_count"], 1)
        self.assertEqual(summary["fmha"]["rejected_count"], 1)
        self.assertFalse(summary["fmha"]["forced_tiled_head128_observed"])
        self.assertEqual(summary["engine_config"]["headDim"], 128)
        self.assertEqual(summary["attention"]["num_kv_heads"], 8)
        self.assertEqual(summary["attention"]["head_dim"], 128)
        self.assertEqual(summary["attention"]["kv_cache_dtype"], "kHALF")
        self.assertFalse(summary["attention"]["use_paged_kv_cache"])
        self.assertEqual(summary["attention"]["max_kv_pool_pages"], 16)
        self.assertEqual(summary["attention"]["tokens_per_page"], 64)
        self.assertEqual(summary["attention"]["spec_decode_type"], 0)
        self.assertTrue(summary["cuda_graph"]["capture_success"])
        self.assertTrue(summary["cuda_graph"]["decoder_capture_success"])
        self.assertEqual(summary["cuda_graph"]["replay_count"], 0)
        self.assertFalse(summary["cuda_graph"]["replay_observed"])
        self.assertEqual(summary["streams"]["worker_stream_counts"], [2])
        self.assertEqual(summary["streams"]["profile_switch_api_calls"][0]["profile"], 0)
        self.assertEqual(summary["streams"]["profile_switch_api_calls"][1]["kind"], "pinned profile initialization call")
        self.assertEqual(
            summary["streams"]["profile_switch_api_call_summary"]["count"], 2
        )
        self.assertEqual(
            summary["streams"]["profile_switch_api_call_summary"]["p50"], 0.389
        )
        self.assertEqual(summary["streams"]["profile_switch_count"], 2)
        self.assertEqual(
            summary["streams"]["profile_transitions"],
            [{"from": 1, "to": 0}, {"from": 0, "to": 1}],
        )
        self.assertEqual(summary["streams"]["same_profile_transition_count"], 0)
        self.assertEqual(summary["streams"]["timed_profile_transition_count"], 2)
        self.assertEqual(
            summary["streams"]["profile_transition_message_intervals_ms"], [660.0]
        )
        self.assertEqual(
            summary["streams"]["profile_transition_message_interval_summary"]["p50"],
            660.0,
        )

    def test_runtime_log_summary_records_forced_fmha_tiled_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "runtime.log"
            log.write_text(
                "[INFO] ContextFMHA: forced tiled head_dim=128 candidate for SM87 (S=768)\n",
                encoding="utf-8",
            )
            summary = summarize_log(log)

        self.assertEqual(summary["fmha"]["forced_tiled_head128_count"], 1)
        self.assertTrue(summary["fmha"]["forced_tiled_head128_observed"])

    def test_runtime_log_summary_records_explicit_graph_disable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "runtime.log"
            log.write_text(
                "[INFO] CUDA graph capture disabled by "
                "EDGELLM_DISABLE_CUDA_GRAPH=1.\n",
                encoding="utf-8",
            )
            summary = summarize_log(log)

        self.assertFalse(summary["cuda_graph"]["requested"])
        self.assertTrue(summary["cuda_graph"]["disabled_by_config"])
        self.assertFalse(summary["cuda_graph"]["capture_success"])
        self.assertFalse(summary["cuda_graph"]["decoder_capture_success"])
        self.assertEqual(summary["cuda_graph"]["replay_count"], 0)
        self.assertFalse(summary["cuda_graph"]["replay_observed"])

    def test_runtime_log_summary_records_explicit_graph_replay_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log = Path(temporary_directory) / "runtime.log"
            log.write_text(
                "\n".join(
                    [
                        "[INFO] CUDA graph enabled",
                        "[INFO] CUDA graph captured successfully.",
                        "[INFO] cudaGraphLaunch decoder graph replay",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_log(log)

        self.assertEqual(summary["cuda_graph"]["replay_count"], 1)
        self.assertTrue(summary["cuda_graph"]["replay_observed"])

    def test_runtime_log_comparison_does_not_claim_tactic_selection(self) -> None:
        left = {
            "engine_config": {"headDim": 128},
            "fmha": {"loaded": [{"index": 8, "sm": 87, "function": "head128"}], "rejected": []},
            "cuda_graph": {"capture_success": True},
        }
        right = {
            "engine_config": {"headDim": 128},
            "fmha": {"loaded": [{"index": 8, "sm": 87, "function": "head128"}], "rejected": []},
            "cuda_graph": {"capture_success": True},
        }
        comparison = compare_summaries(left, right)
        self.assertTrue(comparison["same_loaded_fmha_candidates"])

    def test_profile_contract_rejects_i512_for_observed_735_token_prompt(self) -> None:
        result = validate_profile_contract(
            max_input_len=512,
            max_kv_cache_capacity=768,
            observed_input_tokens=735,
            max_generate_length=32,
        )
        self.assertFalse(result["valid"])
        self.assertIn("max_input_len", result["reasons"][0])

    def test_profile_contract_accepts_i768_k768_for_32_token_output(self) -> None:
        result = validate_profile_contract(
            max_input_len=768,
            max_kv_cache_capacity=768,
            observed_input_tokens=735,
            max_generate_length=32,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["required_kv_capacity"], 767)
        self.assertEqual(result["kv_headroom"], 1)

    def test_estimates_fp16_kv_cache_bytes_for_qwen_gqa_shape(self) -> None:
        self.assertEqual(
            estimate_kv_cache_bytes(
                max_batch_size=1,
                max_kv_cache_capacity=1024,
                num_kv_heads=8,
                head_dim=128,
            ),
            4 * 1024 * 1024,
        )
        self.assertEqual(
            estimate_kv_cache_bytes(
                max_batch_size=1,
                max_kv_cache_capacity=768,
                num_kv_heads=8,
                head_dim=128,
            ),
            3 * 1024 * 1024,
        )

    def test_engine_inspector_comparison_reports_changed_tactic_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.json"
            right = root / "right.json"
            base = {
                "schema_version": "parksight_tensorrt_engine_inspector_v1",
                "engine": {
                    "sha256": "left",
                    "size_bytes": 10,
                    "device_memory_size": 100,
                },
                "layers": [
                    {"Name": "attention_qkv", "LayerType": "MatrixMultiply", "TacticName": "tactic_a", "WorkspaceSize": 32, "AverageMs": 1.5},
                    {"Name": "softmax", "LayerType": "Softmax", "TacticName": "tactic_s"},
                ],
            }
            changed = {
                **base,
                "engine": {"sha256": "right", "size_bytes": 11, "device_memory_size": 120},
                "layers": [
                    {"Name": "attention_qkv", "LayerType": "MatrixMultiply", "TacticName": "tactic_b", "WorkspaceSize": 64, "AverageMs": 2.5},
                    {"Name": "softmax", "LayerType": "Softmax", "TacticName": "tactic_s"},
                ],
            }
            left.write_text(json.dumps(base), encoding="utf-8")
            right.write_text(json.dumps(changed), encoding="utf-8")
            result = compare_inspectors(left, right)

        self.assertEqual(result["layers"]["changed_count"], 1)
        self.assertEqual(result["layers"]["changed"][0]["index"], 0)
        self.assertEqual(result["engine"]["right_device_memory_size"], 120)
        self.assertEqual(result["tactic_visibility"]["left"], "present")
        self.assertEqual(result["tactic_visibility"]["timing"]["left"], "present")
        self.assertEqual(
            result["layers"]["diff_summary"],
            {
                "matched_count": 2,
                "added_count": 0,
                "removed_count": 0,
                "type_changed_count": 0,
                "tactic_changed_count": 1,
                "workspace_changed_count": 1,
                "timing_changed_count": 1,
            },
        )
        self.assertEqual(result["layers"]["operator_count_delta"], {})

    def test_engine_inspector_alignment_reports_fusion_shape_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            left = root / "left.json"
            right = root / "right.json"
            base = {
                "schema_version": "parksight_tensorrt_engine_inspector_v1",
                "engine": {},
                "layers": [
                    {"Name": "qkv", "LayerType": "MatrixMultiply"},
                    {"Name": "softmax", "LayerType": "Softmax"},
                ],
            }
            fused = {
                **base,
                "layers": [
                    {"Name": "qkv", "LayerType": "MatrixMultiply"},
                    {"Name": "softmax_v_projection_fused", "LayerType": "Fused"},
                ],
            }
            left.write_text(json.dumps(base), encoding="utf-8")
            right.write_text(json.dumps(fused), encoding="utf-8")
            result = compare_inspectors(left, right)

        self.assertEqual(result["layers"]["alignment"]["matched_count"], 1)
        self.assertEqual(result["layers"]["alignment"]["added_count"], 1)
        self.assertEqual(result["layers"]["alignment"]["removed_count"], 1)
        self.assertEqual(result["layers"]["changed_count"], 2)
        self.assertEqual(result["layers"]["operator_count_delta"], {"Fused": 1, "Softmax": -1})

    def test_benchmark_comparison_reports_stage_improvement_and_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "int4_awq",
            }
            def summary(metadata: dict[str, object], decode: float, e2e: float) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": metadata,
                    "execution": {
                        "sample_count": 20,
                        "completed_sample_count": 20,
                        "stage_latency_ms": {
                            "decode_ms": {"count": 20, "p50": decode, "p90": decode + 10, "p99": decode + 20},
                            "end_to_end_ms": {"count": 20, "p50": e2e, "p90": e2e + 10, "p99": e2e + 20},
                        },
                        "decode_tokens_per_second": 1000 / decode,
                        "end_to_end_tokens_per_second": 1000 / e2e,
                    },
                }
            baseline.write_text(json.dumps(summary(common, 100, 200)), encoding="utf-8")
            candidate.write_text(
                json.dumps(summary({**common, "precision": "fp16"}, 50, 150)),
                encoding="utf-8",
            )
            result = compare_benchmarks(baseline, candidate)

        self.assertEqual(result["match"]["status"], "mismatch")
        self.assertFalse(result["match"]["comparable"])
        self.assertEqual(
            result["execution"]["stage_latency_ms"]["decode_ms"]["p50"]["improvement"],
            0.5,
        )
        self.assertEqual(
            result["execution"]["decode_tokens_per_second"]["speedup"],
            2.0,
        )

    def test_benchmark_comparison_reports_system_throughput_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "int4_awq",
                "max_batch_size": 4,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "cuda_graph": "enabled",
                "weight_streaming": "disabled",
            }

            def summary(concurrency: int, aggregate: float) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": {
                        **common,
                        "concurrency": concurrency,
                    },
                    "execution": {
                        "sample_count": 4,
                        "completed_sample_count": 4,
                        "stage_latency_ms": {},
                        "aggregate_output_tokens_per_second": aggregate,
                    },
                }

            baseline.write_text(json.dumps(summary(1, 20.0)), encoding="utf-8")
            candidate.write_text(json.dumps(summary(4, 50.0)), encoding="utf-8")
            result = compare_benchmarks(baseline, candidate)

        self.assertTrue(result["match"]["comparable"])
        self.assertEqual(result["run"]["status"], "changed")
        self.assertEqual(result["run"]["changed_fields"]["concurrency"]["candidate"], 4)
        throughput = result["execution"]["aggregate_output_tokens_per_second"]
        self.assertEqual(throughput["speedup"], 2.5)
        self.assertEqual(throughput["improvement"], 1.5)

    def test_benchmark_comparison_checks_optional_visual_image_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "fp16",
                "max_batch_size": 1,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "cuda_graph": "enabled",
                "weight_streaming": "disabled",
                "min_image_tokens": 8,
                "max_image_tokens": 2048,
                "max_image_tokens_per_image": 2048,
            }

            def summary(metadata: dict[str, object]) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": metadata,
                    "execution": {
                        "sample_count": 1,
                        "completed_sample_count": 1,
                        "stage_latency_ms": {},
                    },
                }

            baseline.write_text(json.dumps(summary(common)), encoding="utf-8")
            candidate.write_text(
                json.dumps(summary({**common, "max_image_tokens": 1024})),
                encoding="utf-8",
            )
            result = compare_benchmarks(baseline, candidate)

        self.assertEqual(result["match"]["status"], "mismatch")
        self.assertFalse(result["match"]["comparable"])
        self.assertEqual(
            result["match"]["mismatches"]["max_image_tokens"],
            {"baseline": 2048, "candidate": 1024},
        )

    def test_benchmark_comparison_rejects_paged_kv_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "int4_awq",
                "max_batch_size": 1,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "cuda_graph": "enabled",
                "weight_streaming": "disabled",
            }

            def summary(metadata: dict[str, object]) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": metadata,
                    "execution": {
                        "sample_count": 1,
                        "completed_sample_count": 1,
                        "stage_latency_ms": {},
                    },
                }

            baseline.write_text(json.dumps(summary(common)), encoding="utf-8")
            candidate.write_text(
                json.dumps(summary({**common, "max_kv_pool_pages": 16})),
                encoding="utf-8",
            )
            result = compare_benchmarks(baseline, candidate)

        self.assertEqual(result["match"]["status"], "mismatch")
        self.assertFalse(result["match"]["comparable"])
        self.assertEqual(
            result["match"]["mismatches"]["max_kv_pool_pages"],
            {"baseline": None, "candidate": 16},
        )

    def test_benchmark_comparison_requires_only_visual_engine_to_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "fp16",
                "max_batch_size": 1,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "cuda_graph": "enabled",
                "weight_streaming": "disabled",
                "min_image_tokens": 8,
                "max_image_tokens": 2048,
                "max_image_tokens_per_image": 2048,
            }

            def summary(llm_sha: str, visual_sha: str) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": {
                        **common,
                        "runtime_metadata": {
                            "engines": {
                                "llm": {"sha256": llm_sha},
                                "visual": {"sha256": visual_sha},
                            },
                            "options": {},
                        },
                    },
                    "execution": {
                        "sample_count": 1,
                        "completed_sample_count": 1,
                        "stage_latency_ms": {},
                    },
                }

            baseline.write_text(json.dumps(summary("a" * 64, "b" * 64)), encoding="utf-8")
            candidate.write_text(json.dumps(summary("a" * 64, "c" * 64)), encoding="utf-8")
            result = compare_benchmarks(
                baseline,
                candidate,
                changed_engine_component="visual",
            )

        self.assertEqual(result["component_swap"]["status"], "matched")
        self.assertTrue(result["component_swap"]["comparable"])
        self.assertTrue(result["match"]["comparable"])

    def test_benchmark_comparison_includes_visual_encode_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "fp16",
                "max_batch_size": 1,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "cuda_graph": "enabled",
                "weight_streaming": "disabled",
            }

            def summary(vision_encode_ms: float) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": common,
                    "execution": {
                        "sample_count": 1,
                        "completed_sample_count": 1,
                        "stage_latency_ms": {
                            "vision_encode_ms": {
                                "count": 1,
                                "p50": vision_encode_ms,
                                "p90": vision_encode_ms,
                                "p99": vision_encode_ms,
                            }
                        },
                    },
                }

            baseline.write_text(json.dumps(summary(60.0)), encoding="utf-8")
            candidate.write_text(json.dumps(summary(45.0)), encoding="utf-8")
            result = compare_benchmarks(baseline, candidate)

        comparison = result["execution"]["stage_latency_ms"]["vision_encode_ms"]
        self.assertEqual(comparison["p50"]["baseline"], 60.0)
        self.assertEqual(comparison["p50"]["candidate"], 45.0)
        self.assertAlmostEqual(comparison["p50"]["improvement"], 0.25)

    def test_study_report_comparison_reports_stage_quality_and_resource_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"

            def report(*, e2e: float, decode: float, quality: float, memory: float) -> dict[str, object]:
                return {
                    "study_identity": {
                        "workload_identity": "workload-a",
                        "split": "test",
                        "power_mode": "15W_MODE_0",
                        "repetitions": 3,
                        "runtime_identity": {
                            "backend": "tensorrt_edge_llm",
                            "backend_revision": "edge-a",
                            "model_id": "qwen",
                            "model_revision": "model-a",
                            "adapter_revision": "none",
                            "precision": "int4_awq",
                        },
                    },
                    "quality_metrics": {
                        "json_validity_rate": quality,
                        "risk_level_accuracy": quality,
                        "event_micro_f1": quality,
                    },
                    "performance_metrics": {
                        "backend_completed_sample_count": 60,
                        "tokens_per_second": 10.0,
                        "aggregate_output_tokens_per_end_to_end_second": 8.0,
                        "peak_memory_mb": memory,
                        "stage_latency_ms": {
                            "decode_ms": {"count": 60, "p50": decode, "p90": decode + 10, "p99": decode + 20},
                            "end_to_end_ms": {"count": 60, "p50": e2e, "p90": e2e + 10, "p99": e2e + 20},
                        },
                    },
                }

            baseline.write_text(json.dumps(report(e2e=1000, decode=500, quality=0.8, memory=4000)), encoding="utf-8")
            candidate.write_text(json.dumps(report(e2e=800, decode=400, quality=0.8, memory=4050)), encoding="utf-8")
            result = compare_study_reports(baseline, candidate)

        self.assertTrue(result["match"]["comparable"])
        self.assertAlmostEqual(
            result["execution"]["stage_latency_ms"]["decode_ms"]["p50"]["improvement"],
            0.2,
        )
        self.assertAlmostEqual(
            result["execution"]["stage_latency_ms"]["end_to_end_ms"]["p50"]["speedup"],
            1.25,
        )
        self.assertEqual(
            result["execution"]["stage_latency_ms"]["end_to_end_ms"]["status"],
            "matched",
        )
        self.assertAlmostEqual(
            result["execution"]["performance"]["peak_memory_mb"]["improvement"],
            -0.0125,
        )

    def test_benchmark_comparison_reports_runtime_option_changes_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "model_revision": "model-a",
                "edge_llm_revision": "edge-a",
                "platform": "orin",
                "power_mode": "15W",
                "precision": "int4_awq",
            }

            def summary(options: dict[str, object]) -> dict[str, object]:
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": {
                        **common,
                        "runtime_metadata": {
                            "engines": {
                                "llm": {"sha256": "a" * 64},
                                "visual": {"sha256": "b" * 64},
                            },
                            "options": options,
                        },
                    },
                    "execution": {
                        "sample_count": 1,
                        "completed_sample_count": 1,
                        "stage_latency_ms": {
                            "end_to_end_ms": {
                                "count": 1,
                                "p50": 100,
                                "p90": 100,
                                "p99": 100,
                            }
                        },
                    },
                }

            baseline.write_text(
                json.dumps(summary({"pin": "default"})), encoding="utf-8"
            )
            candidate.write_text(
                json.dumps(summary({"pin": "enabled"})), encoding="utf-8"
            )
            result = compare_benchmarks(baseline, candidate)

        self.assertEqual(result["runtime"]["status"], "changed")
        self.assertTrue(result["runtime"]["same_engine"])
        self.assertEqual(
            result["runtime"]["changed_options"]["pin"],
            {"baseline": "default", "candidate": "enabled"},
        )

    def test_benchmark_comparison_does_not_assume_missing_engine_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            common = {
                "schema_version": "parksight_tensorrt_benchmark_v1",
                "metadata": {
                    "runtime_metadata": {"engines": {}, "options": {}},
                },
                "execution": {
                    "stage_latency_ms": {},
                },
            }
            baseline.write_text(json.dumps(common), encoding="utf-8")
            candidate.write_text(json.dumps(common), encoding="utf-8")
            result = compare_benchmarks(baseline, candidate)

        self.assertEqual(result["runtime"]["status"], "unverified")
        self.assertIsNone(result["runtime"]["same_engine"])

    def test_benchmark_comparison_repetition_gate_requires_each_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"

            def summary(values: list[tuple[float, float]]) -> dict[str, object]:
                repetitions = {
                    str(index): {
                        "sample_count": 20,
                        "stage_latency_ms": {
                            "end_to_end_ms": {
                                "count": 20,
                                "p50": p50,
                                "p90": p90,
                                "p99": p90,
                            }
                        },
                    }
                    for index, (p50, p90) in enumerate(values, start=1)
                }
                return {
                    "schema_version": "parksight_tensorrt_benchmark_v1",
                    "metadata": {},
                    "execution": {
                        "sample_count": 60,
                        "completed_sample_count": 60,
                        "repetitions": repetitions,
                        "stage_latency_ms": {},
                    },
                }

            baseline.write_text(
                json.dumps(summary([(100, 120), (100, 120), (100, 120)])),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(summary([(80, 110), (95, 110), (80, 110)])),
                encoding="utf-8",
            )
            result = compare_benchmarks(baseline, candidate)

        gate = result["execution"]["repetition_gate"]
        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["eligible"])
        self.assertTrue(gate["per_repetition"]["1"]["eligible"])
        self.assertFalse(gate["per_repetition"]["2"]["p50_pass"])


def _study_report(
    *,
    study_id: str,
    repetitions: int,
    p50: float,
    p90: float,
    quality: float,
    records: int,
    json_validity: float = 1.0,
) -> dict[str, object]:
    return {
        "study_identity": {
            "study_id": study_id,
            "workload_identity": "parking_risk_v1@sha256:test",
            "runtime_identity": {
                "backend": "tensorrt_edge_llm",
                "backend_revision": "edge-a",
                "model_id": "qwen",
                "model_revision": "model-a",
                "adapter_revision": "adapter-a",
                "precision": "int4_awq",
            },
            "split": "test",
            "repetitions": repetitions,
            "power_mode": "15W_MODE_0",
        },
        "quality_metrics": {
            "json_validity_rate": json_validity,
            "risk_level_accuracy": quality,
            "event_micro_f1": quality,
        },
        "performance_metrics": {
            "backend_completed_sample_count": records,
            "successful_sample_count": int(records * json_validity),
            "stage_latency_ms": {
                "end_to_end_ms": {"p50": p50, "p90": p90}
            },
            "peak_memory_mb": 5000,
        },
        "records": [{} for _ in range(records)],
    }


def _soak_report() -> dict[str, object]:
    return {
        "study_identity": {
            "workload_identity": "parking_risk_v1@sha256:test",
            "runtime_identity": {
                "backend": "tensorrt_edge_llm",
                "backend_revision": "edge-a",
                "model_id": "qwen",
                "model_revision": "model-a",
                "adapter_revision": "adapter-a",
                "precision": "int4_awq",
            },
            "split": "test",
            "power_mode": "15W_MODE_0",
        },
        "failure_summary": {},
        "records": [{} for _ in range(100)],
    }


def _jetson_runtime_summary(
    report: dict[str, object], *, memory: float, records: int | None = None
) -> dict[str, object]:
    study_identity = report["study_identity"]
    sample_count = len(report["records"]) if records is None else records
    return {
        "schema_version": "parksight_jetson_runtime_summary_v1",
        "study_identity": study_identity,
        "runtime_execution": {"record_count": sample_count},
        "jetson_telemetry": {
            "ram_used_mb": {"maximum": memory},
            "vdd_in_w": {"mean": 12.0},
            "gpu_temperature_c": {"maximum": 60.0},
            "swap_used_mb": {"maximum": 100.0},
            "gpu_utilization_percent": {"mean": 98.0},
        },
        "evidence_sources": {"study_report": "study.json", "tegrastats_log": "tegrastats.log"},
    }


if __name__ == "__main__":
    unittest.main()

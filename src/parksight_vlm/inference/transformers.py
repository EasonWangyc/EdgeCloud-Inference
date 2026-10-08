"""Transformers Runtime Adapter 与 Qwen3-VL 参考后端。"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

from parksight_vlm.workload import FrozenWorkload

from .images import load_workload_image

from .runtime import (
    ResourceSnapshot,
    RiskRuntime,
    RuntimeDependencyError,
    RuntimeGeneration,
    RuntimeIdentity,
    StageTimings,
)


class TransformersBackend(Protocol): # 类似cpp中类中的虚函数
    """由模型集成层实现的可执行 Transformers 接口。"""

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """运行模型并返回原始输出和实测事实。"""


class HuggingFaceQwen3VlBackend:
    """遵循上游 Transformers chat API 的 Qwen3-VL 延迟加载后端。"""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        device_map: str | None = "auto",
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        adapter_path: str | None = None,
        profile_stages: bool = False,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._device_map = device_map
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._adapter_path = adapter_path
        self._profile_stages = profile_stages
        self._processor = None
        self._model = None # 构造时不立即加载模型
        self._torch = None

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        preprocess_start = time.perf_counter()
        image = load_workload_image(image_path, workload)
        messages = build_qwen3_vl_chat_messages(image=image, workload=workload)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt", # 张量字典
        )
        inputs.pop("token_type_ids", None)
        if hasattr(inputs, "to") and hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        generate_start = time.perf_counter()
        profiler = (
            _ForwardPhaseProfiler(self._model, self._torch)
            if self._profile_stages
            else None
        )
        first_token_probe = (
            _FirstTokenTimingProbe(self._torch, generation_start=generate_start)
            if self._profile_stages
            else None
        )
        profile_context = profiler if profiler is not None else nullcontext()
        with self._torch.inference_mode(), profile_context: # 纯推理，不计算梯度也不保存反向传播状态
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": workload.generation.max_new_tokens,
                "do_sample": workload.generation.do_sample,
            }
            if first_token_probe is not None:
                # LogitsProcessor 在首个 LM forward 完成、首 token 被选出之前运行。
                # 这比用完整响应返回时间近似 TTFT 更接近真实首 token 时刻。
                generate_kwargs["logits_processor"] = [first_token_probe]
            generated_ids = self._model.generate(**inputs, **generate_kwargs)
        model_generate_ms = (time.perf_counter() - generate_start) * 1000.0
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] # 清除输入prompt对应token
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode( # 由token得到文字
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output_tokens = sum(len(token_ids) for token_ids in generated_ids_trimmed)

        peak_memory_mb = None
        if self._torch.cuda.is_available():
            peak_memory_mb = self._torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return RuntimeGeneration(
            raw_output=output_text,
            stage_timings=StageTimings(
                preprocess_ms=preprocess_ms,
                vision_encode_ms=(
                    profiler.vision_encode_ms if profiler is not None else None
                ),
                model_generate_ms=model_generate_ms,
                prefill_ms=profiler.prefill_ms if profiler is not None else None,
                decode_ms=profiler.decode_ms if profiler is not None else None,
                time_to_first_token_ms=(
                    first_token_probe.elapsed_ms
                    if first_token_probe is not None
                    else None
                ),
            ),
            resource_snapshot=ResourceSnapshot(peak_memory_mb=peak_memory_mb),
            output_tokens=output_tokens,
            decode_tokens=(
                profiler.decode_token_count(output_tokens=output_tokens, batch_size=len(generated_ids_trimmed))
                if profiler is not None else None
            ),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeDependencyError(
                "torch and transformers with Qwen3-VL support are required"
            ) from error
        _require_cuda_architecture(torch)
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            revision=self._model_revision,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id,
            revision=self._model_revision,
            device_map=self._device_map,
            dtype=self._dtype,
            attn_implementation=self._attn_implementation,
        )
        if self._adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise RuntimeDependencyError(
                    "peft is required when a Transformers adapter_path is configured"
                ) from error
            self._model = PeftModel.from_pretrained(
                self._model,
                self._adapter_path,
            )
        self._model.eval()


def build_qwen3_vl_chat_messages(
    *,
    image: Any,
    workload: FrozenWorkload,
) -> list[dict[str, Any]]:
    """按 Qwen3-VL Processor 要求构造统一的多模态 content 列表。"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": workload.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": workload.render_user_prompt()},
            ],
        },
    ]


def _require_cuda_architecture(torch_module: Any) -> None:
    """拒绝无法在已检测 GPU 上执行 kernel 的 CUDA wheel。"""
    if not torch_module.cuda.is_available():
        return
    major, minor = torch_module.cuda.get_device_capability()
    required_architecture = f"sm_{major}{minor}"
    supported_architectures = tuple(torch_module.cuda.get_arch_list())
    if supported_architectures and required_architecture not in supported_architectures:
        # 桌面 GPU 可以通过兼容 SASS/PTX 正常运行，即使 get_arch_list()
        # 没有逐项列出设备的小版本（例如 Ada sm_89 使用 sm_86 兼容代码）。
        # 因此用真实 CUDA kernel 作为最终判据；Jetson 不兼容 wheel 仍会在此失败。
        try:
            probe = torch_module.ones(1, device="cuda")
            probe.add_(1)
            torch_module.cuda.synchronize()
        except Exception as error:
            supported_text = ", ".join(supported_architectures)
            raise RuntimeDependencyError(
                f"installed torch {torch_module.__version__} does not include CUDA "
                f"kernels for {required_architecture}; supported architectures: "
                f"{supported_text}"
            ) from error


class _ForwardPhaseProfiler:
    """用模块 forward hook 细分视觉、prefill 与 decode 时延。

    该诊断会在 hook 边界同步 CUDA，因此只用于独立 profiling study，不能替代
    未插桩基线的端到端时延。语言模型第一次 forward 记为 prefill，后续 forward
    记为 decode；找不到对应模块时保留 ``None``，不伪造阶段数据。
    """

    def __init__(
        self,
        model: Any,
        torch_module: Any,
    ) -> None:
        self._torch = torch_module
        self._handles: list[Any] = []
        self._starts: dict[str, float] = {}
        self._vision_calls: list[float] = []
        self._language_calls: list[float] = []
        modules = dict(model.named_modules())
        self._visual_module = _find_profile_module(modules, "visual")
        self._language_module = _find_profile_module(modules, "language_model")

    def __enter__(self) -> "_ForwardPhaseProfiler":
        if self._visual_module is not None:
            self._register(self._visual_module, "visual", self._vision_calls)
        if self._language_module is not None:
            self._register(self._language_module, "language", self._language_calls)
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register(self, module: Any, name: str, samples: list[float]) -> None:
        def before(*_: Any) -> None:
            self._synchronize()
            self._starts[name] = time.perf_counter()

        def after(*_: Any) -> None:
            self._synchronize()
            started = self._starts.pop(name, None)
            if started is not None:
                samples.append((time.perf_counter() - started) * 1000.0)

        self._handles.append(module.register_forward_pre_hook(before))
        self._handles.append(module.register_forward_hook(after))

    def _synchronize(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    @property
    def vision_encode_ms(self) -> float | None:
        return sum(self._vision_calls) if self._vision_calls else None

    @property
    def prefill_ms(self) -> float | None:
        return self._language_calls[0] if self._language_calls else None

    @property
    def decode_ms(self) -> float | None:
        return sum(self._language_calls[1:]) if len(self._language_calls) > 1 else None

    def decode_token_count(self, *, output_tokens: int, batch_size: int) -> int | None:
        """Count observed single-sequence decode forwards, excluding prefill.

        Report only when the observed forwards agree with normal autoregressive
        generation. Other batching or forward patterns retain unknown counts.
        """
        if not self._language_calls or batch_size != 1 or output_tokens < 1:
            return None
        observed = len(self._language_calls) - 1
        return observed if observed == output_tokens - 1 else None


class _FirstTokenTimingProbe:
    """在首个 logits 可用时记录近似 TTFT。"""

    def __init__(self, torch_module: Any, *, generation_start: float) -> None:
        self._torch = torch_module
        self._generation_start = generation_start
        self.elapsed_ms: float | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        del input_ids
        if self.elapsed_ms is None:
            if self._torch.cuda.is_available():
                self._torch.cuda.synchronize()
            self.elapsed_ms = (time.perf_counter() - self._generation_start) * 1000.0
        return scores


def _find_profile_module(modules: dict[str, Any], leaf_name: str) -> Any | None:
    """选择最浅的匹配模块，兼容 ``model.visual`` 等上游命名。"""
    matches = [
        (name.count("."), name, module)
        for name, module in modules.items()
        if name == leaf_name or name.endswith(f".{leaf_name}")
    ]
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], item[1]))[2]


class TransformersRuntime(RiskRuntime):
    """记录 Transformers 后端执行事实的 Adapter。"""

    def __init__(
        self,
        *,
        data_root: Path,
        backend: TransformersBackend,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str = "none",
        precision: str = "bf16",
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="transformers",
                backend_revision=backend_revision,
                model_id=model_id,
                model_revision=model_revision,
                adapter_revision=adapter_revision,
                precision=precision,
            ),
        )
        self._backend = backend

    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        return self._backend.generate(image_path=image_path, workload=workload)

"""Runtime Adapter 与不可变推理执行记录。"""

from .edge_llm import (
    EdgeLlmBackend,
    EdgeLlmHttpBackend,
    EdgeLlmRuntime,
    OpenAICompatibleHttpBackend,
)
from .runtime import (
    InferenceRecord,
    ResourceSnapshot,
    RiskRuntime,
    RuntimeDependencyError,
    RuntimeFailure,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeIdentity,
    RuntimeRefusalError,
    RuntimeUnsupportedError,
    StageTimings,
)
from .transformers import (
    HuggingFaceQwen3VlBackend,
    TransformersBackend,
    TransformersRuntime,
)
from .vllm import VllmHttpBackend, VllmRuntime
from .routing import (
    EdgeCloudRouterRuntime,
    HttpHealthRoutingSignalsProvider,
    RoutingPolicy,
    RoutingSignals,
    RoutingSignalsProvider,
    StaticRoutingSignalsProvider,
)

__all__ = [
    "EdgeLlmBackend",
    "EdgeLlmHttpBackend",
    "EdgeLlmRuntime",
    "OpenAICompatibleHttpBackend",
    "InferenceRecord",
    "ResourceSnapshot",
    "RiskRuntime",
    "RuntimeDependencyError",
    "RuntimeFailure",
    "RuntimeFailureCategory",
    "RuntimeGeneration",
    "RuntimeIdentity",
    "RuntimeRefusalError",
    "RuntimeUnsupportedError",
    "StageTimings",
    "HuggingFaceQwen3VlBackend",
    "TransformersBackend",
    "TransformersRuntime",
    "VllmHttpBackend",
    "VllmRuntime",
    "EdgeCloudRouterRuntime",
    "HttpHealthRoutingSignalsProvider",
    "RoutingPolicy",
    "RoutingSignals",
    "RoutingSignalsProvider",
    "StaticRoutingSignalsProvider",
]

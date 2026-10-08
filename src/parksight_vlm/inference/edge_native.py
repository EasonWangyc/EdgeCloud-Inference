"""Select a source-bound native binding without replacing the deployed module."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import sha256_file


def binding_identity(directory: Path, expected_sha256: str) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("native binding SHA-256 must be 64 lowercase hexadecimal characters")
    candidates = list(directory.resolve().glob("_edgellm_runtime*.so"))
    if len(candidates) != 1:
        raise ValueError("native binding directory must contain exactly one _edgellm_runtime .so")
    path = candidates[0].resolve()
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError("native binding SHA-256 does not match the requested identity")
    return {"path": str(path), "sha256": actual}


def load_native_binding(directory: Path, expected_sha256: str) -> Any:
    identity = binding_identity(directory, expected_sha256)
    name = "_edgellm_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve() != Path(identity["path"]):
            raise RuntimeError("another native binding is already loaded; start a fresh process")
        if getattr(existing, "_parksight_loaded_sha256", None) != expected_sha256:
            raise RuntimeError("loaded native binding identity is unverified; start a fresh process")
        return existing
    spec = importlib.util.spec_from_file_location(name, identity["path"])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create native binding loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if sha256_file(Path(identity["path"])) != expected_sha256:
            raise RuntimeError("native binding changed during import; start a fresh process")
        module._parksight_loaded_sha256 = expected_sha256
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def construct_with_binding(engine_module: Any, runtime_module: Any, **options: Any) -> Any:
    """Upstream's path loader otherwise imports the validated extension twice.

    Use only during single-threaded initialization, before accepting HTTP requests.
    """
    original = engine_module._import_runtime
    engine_module._import_runtime = lambda: runtime_module
    try:
        return engine_module.LLM(**options)
    finally:
        engine_module._import_runtime = original

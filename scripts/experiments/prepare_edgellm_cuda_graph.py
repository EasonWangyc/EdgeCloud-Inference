"""Prepare a source-bound Python Graph switch for an isolated server copy."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def cuda_graph_source(source: bytes, expected_sha256: str) -> bytes:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("server source SHA-256 changed")
    text = source.decode("utf-8")
    anchor = "        self._runtime.capture_decoding_cuda_graph()"
    if "EDGELLM_DISABLE_CUDA_GRAPH" in text:
        raise ValueError("Graph switch already exists")
    if text.count(anchor) != 1 or text.count("import os") != 1:
        raise ValueError("unsupported server Graph capture layout")
    suffix = text[text.index(anchor) + len(anchor):]
    newline = "\r\n" if suffix.startswith("\r\n") else "\n"
    replacement = '''        if os.environ.get("EDGELLM_DISABLE_CUDA_GRAPH") == "1":
            logger.info("CUDA graph capture disabled by EDGELLM_DISABLE_CUDA_GRAPH=1.")
        else:
            self._runtime.capture_decoding_cuda_graph()'''.replace("\n", newline)
    return text.replace(anchor, replacement, 1).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata.exists():
        parser.error("patch evidence already exists")
    source = args.source.read_bytes()
    updated = cuda_graph_source(source, args.source_sha256)
    patch = "".join(difflib.unified_diff(
        source.decode().splitlines(keepends=True), updated.decode().splitlines(keepends=True),
        fromfile="a/experimental/server/engine.py", tofile="b/experimental/server/engine.py",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as file:
        file.write(patch)
    with metadata.open("x") as file:
        json.dump({
            "source_sha256": args.source_sha256,
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "resulting_source_sha256": hashlib.sha256(updated).hexdigest(),
            "source_relative_path": "experimental/server/engine.py",
            "switch": "EDGELLM_DISABLE_CUDA_GRAPH=1",
            "scope": "Python capture entry only; apply to isolated copy; no native rebuild or GPU proof",
            "status": "prepared",
        }, file, indent=2)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

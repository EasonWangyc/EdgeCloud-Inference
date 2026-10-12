"""Prepare a source-bound opt-in deferred execution-context allocation experiment."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def context_allocation_patch(source: bytes, expected_sha256: str) -> tuple[str, str]:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("runtime source SHA-256 changed")
    text = source.decode()
    if "EDGELLM_DEFER_CONTEXT_ALLOCATION" in text:
        raise ValueError("context allocation experiment already exists")
    start = '    int64_t sharedContextMemorySize = std::max(baseContextMemorySize, strategyContextMemorySize);'
    end = '        static_cast<size_t>(strategyContextMemorySize));'
    growth = '    if (requiredSharedContextMemorySize > sharedContextMemorySize)'
    if any(text.count(anchor) != 1 for anchor in (start, end, growth, '#include <string>')):
        raise ValueError("unsupported execution context allocation layout")
    begin = text.index(start)
    finish = text.index(end, begin) + len(end)
    block = text[begin:finish]
    newline = "\r\n" if "\r\n" in block else "\n"
    declaration, sep, allocation = block.partition(newline)
    if not sep or allocation.count('mSharedExecContextMemory = rt::Tensor') != 1:
        raise ValueError("unsupported preallocation block")
    option = '''    char const* contextOption = std::getenv("EDGELLM_DEFER_CONTEXT_ALLOCATION");
    if (contextOption != nullptr && std::string_view(contextOption) != "0"
        && std::string_view(contextOption) != "1")
    {
        throw std::invalid_argument("EDGELLM_DEFER_CONTEXT_ALLOCATION must be 0 or 1");
    }
    bool const deferContextAllocation = contextOption != nullptr && std::string_view(contextOption) == "1";
    if (deferContextAllocation && hasDraft)
    {
        throw std::invalid_argument("Deferred context allocation is validated only for vanilla decoding");
    }
'''.replace("\n", newline)
    wrapped = newline.join("    " + line for line in allocation.split(newline))
    replacement = option + declaration + newline + '    if (!deferContextAllocation)' + newline + '    {' + newline
    replacement += wrapped + newline + '    }'
    updated = text[:begin] + replacement + text[finish:]
    updated = updated.replace(growth, '    if (deferContextAllocation || requiredSharedContextMemorySize > sharedContextMemorySize)', 1)
    updated = updated.replace('#include <string>', '#include <string>' + newline + '#include <string_view>'
                              + newline + '#include <cstdlib>' + newline + '#include <stdexcept>', 1)
    log_anchor = '        sharedContextMemorySize = requiredSharedContextMemorySize;'
    if updated.count(log_anchor) != 1:
        raise ValueError("unsupported shared context allocation layout")
    updated = updated.replace(log_anchor, log_anchor + newline
                              + '        LOG_INFO("Allocating final shared execution context: %zu bytes (deferred=%d)",'
                              + newline + '            static_cast<size_t>(sharedContextMemorySize), deferContextAllocation ? 1 : 0);', 1)
    patch = "".join(difflib.unified_diff(text.splitlines(keepends=True), updated.splitlines(keepends=True),
                                      fromfile="a/cpp/runtime/llmInferenceRuntime.cpp",
                                      tofile="b/cpp/runtime/llmInferenceRuntime.cpp"))
    return patch, hashlib.sha256(updated.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or metadata.exists():
        parser.error("patch evidence already exists")
    patch, result_sha = context_allocation_patch(args.source.read_bytes(), args.source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as file:
        file.write(patch)
    with metadata.open("x") as file:
        json.dump({"source_sha256": args.source_sha256, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                   "resulting_source_sha256": result_sha, "source_relative_path": "cpp/runtime/llmInferenceRuntime.cpp",
                   "switch": "EDGELLM_DEFER_CONTEXT_ALLOCATION=1", "status": "prepared",
                   "scope": "Vanilla decoding only; preserve default allocation order; no GPU proof"}, file, indent=2)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""审计固定 Edge-LLM checkout 的 paged KV cache 源码边界。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


class KVCacheAuditError(RuntimeError):
    """Edge-LLM checkout 不满足源码审计条件。"""


_CHECKS: tuple[tuple[str, str, str], ...] = (
    (
        "builder_paged_kv_pool_option",
        "examples/llm/llm_build.cpp",
        r"maxKVPoolPages|max_kv_pool_pages|kvPoolPages",
    ),
    (
        "paged_kv_runtime_symbols",
        "cpp",
        r"KVCacheType::kPAGED|isPagedKVCache|usePagedKVCache|use_paged_kv_cache",
    ),
    (
        "paged_kv_pool_symbols",
        "cpp",
        r"maxKVPoolPages|max_kv_pool_pages|tokensPerPage|tokens_per_page|pagePool|page_pool",
    ),
    (
        "paged_xqa_abi_symbols",
        "cpp",
        r"pageList|page_list|tokensPerPage|tokens_per_page",
    ),
    (
        "attention_plugin_paged_binding_symbols",
        "cpp/plugins/attentionPlugin/attentionPlugin.cpp",
        r"pageList|page_list|tokensPerPage|tokens_per_page|pagedKV|paged_kv",
    ),
    (
        "runtime_page_table_update",
        "cpp/runtime/state/kvPageTable.cpp",
        r"KVPageTable::setRows\s*\(",
    ),
    (
        "runtime_page_table_compact",
        "cpp/runtime/state/kvPageTable.cpp",
        r"KVPageTable::compactRows\s*\(",
    ),
    (
        "runtime_page_table_upload",
        "cpp/runtime/state/kvPageTable.cpp",
        r"KVPageTable::upload\s*\(",
    ),
)

_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hpp"}


def audit_kv_cache_source(
    *, edge_llm_root: Path, expected_revision: str | None = None
) -> dict[str, Any]:
    """Inspect known v0.9.1 source points without modifying the checkout."""
    root = edge_llm_root.resolve()
    if not root.is_dir():
        raise KVCacheAuditError(f"Edge-LLM root does not exist: {root}")

    revision = _git_output(root, ("rev-parse", "HEAD"))
    if expected_revision is not None and revision != expected_revision:
        raise KVCacheAuditError(
            "Edge-LLM revision mismatch: "
            f"expected={expected_revision}, actual={revision}"
        )

    source_cache: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    for check_id, relative_path, pattern in _CHECKS:
        matches = _find_matches(
            root=root,
            relative_path=relative_path,
            pattern=pattern,
            source_cache=source_cache,
        )
        checks.append(
            {
                "id": check_id,
                "path": relative_path,
                "pattern": pattern,
                "matched": bool(matches),
                "matches": matches[:20],
            }
        )

    # v0.9.1 ships paged-KV data structures and kernels, but the attention
    # plugin can still pin the runtime route to contiguous KV. Keep this
    # source-level fact separate from generic symbol detection so an ABI
    # symbol is not mistaken for a reachable build/runtime path.
    attention_plugin_path = root / "cpp/plugins/attentionPlugin/attentionPlugin.cpp"
    hardcoded_disabled_matches = (
        _find_matches(
            root=root,
            relative_path="cpp/plugins/attentionPlugin/attentionPlugin.cpp",
            pattern=r"bool\s+const\s+usePagedKVCache\s*=\s*false",
            source_cache=source_cache,
        )
        if attention_plugin_path.is_file()
        else []
    )
    checks.append(
        {
            "id": "attention_plugin_hardcodes_paged_kv_disabled",
            "path": "cpp/plugins/attentionPlugin/attentionPlugin.cpp",
            "pattern": r"bool\s+const\s+usePagedKVCache\s*=\s*false",
            "matched": bool(hardcoded_disabled_matches),
            "matches": hardcoded_disabled_matches[:20],
            "optional": True,
        }
    )

    by_id = {item["id"]: item for item in checks}
    builder_has_pool_option = by_id["builder_paged_kv_pool_option"]["matched"]
    runtime_has_paged_symbols = by_id["paged_kv_runtime_symbols"]["matched"]
    pool_has_paged_symbols = by_id["paged_kv_pool_symbols"]["matched"]
    xqa_has_page_symbols = by_id["paged_xqa_abi_symbols"]["matched"]
    plugin_has_paged_bindings = by_id["attention_plugin_paged_binding_symbols"]["matched"]
    runtime_has_page_table_operations = all(
        by_id[check_id]["matched"]
        for check_id in (
            "runtime_page_table_update",
            "runtime_page_table_compact",
            "runtime_page_table_upload",
        )
    )
    attention_plugin_hardcodes_disabled = by_id[
        "attention_plugin_hardcodes_paged_kv_disabled"
    ]["matched"]

    complete_source_path = all(
        (
            builder_has_pool_option,
            runtime_has_paged_symbols,
            pool_has_paged_symbols,
            xqa_has_page_symbols,
            plugin_has_paged_bindings,
            runtime_has_page_table_operations,
        )
    ) and not attention_plugin_hardcodes_disabled

    if not complete_source_path and (
        attention_plugin_hardcodes_disabled
        or builder_has_pool_option
        or runtime_has_paged_symbols
        or pool_has_paged_symbols
        or xqa_has_page_symbols
        or plugin_has_paged_bindings
        or runtime_has_page_table_operations
    ):
        conclusion = "paged_kv_not_wired_in_checkout"
    elif complete_source_path:
        conclusion = "paged_kv_path_exposed_requires_runtime_validation"
    else:
        conclusion = "paged_kv_not_detected_in_checkout"

    return {
        "schema_version": "parksight_edgellm_kv_cache_audit_v3",
        "edge_llm_root": str(root),
        "edge_llm_revision": revision,
        "source_fingerprint": _source_fingerprint(root),
        "checks": checks,
        "summary": {
            "builder_has_paged_kv_pool_option": builder_has_pool_option,
            "runtime_has_paged_kv_symbols": runtime_has_paged_symbols,
            "pool_has_paged_kv_symbols": pool_has_paged_symbols,
            "xqa_has_page_symbols": xqa_has_page_symbols,
            "attention_plugin_has_paged_bindings": plugin_has_paged_bindings,
            "runtime_has_page_table_operations": runtime_has_page_table_operations,
            "complete_source_path": complete_source_path,
            "attention_plugin_hardcodes_paged_kv_disabled": attention_plugin_hardcodes_disabled,
            "conclusion": conclusion,
        },
        "evidence_boundary": (
            "Source markers describe available code paths only; they do not prove that a "
            "serialized engine or a running request uses paged KV. Confirm runtime logs, "
            "kernel arguments and Nsight traces separately."
        ),
    }


def _source_fingerprint(root: Path) -> dict[str, Any]:
    """Identify the source being audited even when HEAD has local patches."""
    candidates = sorted(
        (
            path
            for directory in (root / "cpp", root / "examples/llm")
            for path in directory.rglob("*")
            if path.is_file()
            and (path.suffix.lower() in _SOURCE_SUFFIXES or path.name == "CMakeLists.txt")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in candidates:
        if not path.resolve().is_relative_to(root):
            raise KVCacheAuditError(f"source file escapes Edge-LLM root: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        try:
            size = path.stat().st_size
            digest.update(size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as error:
            raise KVCacheAuditError(f"cannot fingerprint source file: {path}") from error
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(candidates),
        "scope": "cpp and examples/llm source files and CMakeLists.txt",
    }


def _find_matches(
    *, root: Path, relative_path: str, pattern: str, source_cache: dict[str, str]
) -> list[dict[str, Any]]:
    path = root / relative_path
    candidates = [path] if path.is_file() else _source_files(path)
    if not candidates:
        return []

    compiled = re.compile(pattern)
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        cache_key = str(candidate)
        try:
            text = source_cache.setdefault(
                cache_key, candidate.read_text(encoding="utf-8", errors="replace")
            )
        except OSError as error:
            raise KVCacheAuditError(f"cannot read source file: {candidate}") from error
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = compiled.search(line)
            if match is not None:
                matches.append(
                    {
                        "path": str(candidate.relative_to(root)),
                        "line": line_number,
                        "match": match.group(0),
                    }
                )
    return matches


def _source_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _SOURCE_SUFFIXES
    )


def _git_output(root: Path, arguments: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise KVCacheAuditError(f"git command failed to start: {error}") from error
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise KVCacheAuditError(
            f"git command failed with exit code {result.returncode}: {details}"
        )
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", required=True, type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = audit_kv_cache_source(
            edge_llm_root=args.edge_llm_root,
            expected_revision=args.expected_revision,
        )
    except KVCacheAuditError as error:
        parser.error(str(error))

    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

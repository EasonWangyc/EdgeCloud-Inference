"""在临时 git worktree 中验证 TensorRT Edge-LLM 候选补丁链。"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from parksight_vlm.tensorrt import TensorRTValidationError, sha256_file as _sha256_file


class PatchChainError(RuntimeError):
    """补丁链或 Edge-LLM checkout 不满足校验条件。"""


def sha256_file(path: Path) -> str:
    try:
        return _sha256_file(path)
    except TensorRTValidationError as error:
        raise PatchChainError(f"cannot read patch: {path}") from error


def check_patch_chain(
    *,
    edge_llm_root: Path,
    patches: Sequence[Path],
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Apply a patch sequence in a temporary worktree and return an audit report."""
    root = edge_llm_root.resolve()
    if not root.is_dir():
        raise PatchChainError(f"Edge-LLM root does not exist: {root}")
    if not patches:
        raise PatchChainError("at least one patch is required")

    normalized_patches = tuple(path.resolve() for path in patches)
    missing = [str(path) for path in normalized_patches if not path.is_file()]
    if missing:
        raise PatchChainError(f"patch files do not exist: {missing}")

    revision = _git_output(root, ("rev-parse", "HEAD"))
    if expected_revision is not None and revision != expected_revision:
        raise PatchChainError(
            "Edge-LLM revision mismatch: "
            f"expected={expected_revision}, actual={revision}"
        )

    patch_records = [
        {
            "path": str(path),
            "name": path.name,
            "sha256": sha256_file(path),
        }
        for path in normalized_patches
    ]

    with tempfile.TemporaryDirectory(prefix="edgellm-patch-chain-") as temporary_dir:
        worktree = Path(temporary_dir) / "checkout"
        _run_git(root, ("worktree", "add", "--detach", str(worktree), revision))
        try:
            for index, patch in enumerate(normalized_patches, start=1):
                _run_git(
                    worktree,
                    ("apply", "--check", "--recount", "--unidiff-zero", str(patch)),
                    context=f"patch {index} {patch.name} check",
                )
                _run_git(
                    worktree,
                    ("apply", "--recount", "--unidiff-zero", str(patch)),
                    context=f"patch {index} {patch.name} apply",
                )
        finally:
            _run_git(
                root,
                ("worktree", "remove", "--force", str(worktree)),
                context="temporary worktree cleanup",
            )

    return {
        "schema_version": "parksight_tensorrt_patch_chain_v1",
        "status": "succeeded",
        "edge_llm_root": str(root),
        "edge_llm_revision": revision,
        "patches": patch_records,
        "applied_in_order": [record["name"] for record in patch_records],
    }


def _git_output(root: Path, arguments: tuple[str, ...]) -> str:
    result = _run_git(root, arguments)
    return result.stdout.strip()


def _run_git(
    root: Path,
    arguments: tuple[str, ...],
    *,
    context: str = "git command",
) -> subprocess.CompletedProcess[str]:
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
        raise PatchChainError(f"{context} failed to start: {error}") from error
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise PatchChainError(
            f"{context} failed with exit code {result.returncode}: {details}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", required=True, type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--patch", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = check_patch_chain(
            edge_llm_root=args.edge_llm_root,
            patches=args.patch,
            expected_revision=args.expected_revision,
        )
    except PatchChainError as error:
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

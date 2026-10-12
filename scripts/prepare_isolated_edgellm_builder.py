"""Rebuild an Edge-LLM LLM builder in a new directory from a pinned SDK source.

Only builderUtils.cpp is recompiled. The source checkout, its archive, and its
llm_build executable are never edited. Run on the target Jetson with its SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flags(path: Path) -> list[str]:
    variables = {}
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator:
            variables[name.strip()] = value.strip()
    return [
        token
        for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")
        for token in shlex.split(variables[name])
    ]


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    with log.open("x") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def prepare(
    *, edge_root: Path, patch: Path, output: Path,
    expected_revision: str, expected_source_sha256: str,
) -> dict:
    edge_root = edge_root.resolve()
    patch = patch.resolve()
    output = output.resolve()
    if output == edge_root or edge_root in output.parents:
        raise ValueError("output must be outside the Edge-LLM source checkout")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(edge_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != expected_revision:
        raise ValueError(f"SDK revision mismatch: {actual_revision}")
    source = edge_root / "cpp/builder/builderUtils.cpp"
    original_archive = edge_root / "build/cpp/libedgellmBuilder.a"
    original_builder = edge_root / "build/examples/llm/llm_build"
    original = {
        "source": sha256_file(source),
        "archive": sha256_file(original_archive),
        "builder": sha256_file(original_builder),
    }
    if original["source"] != expected_source_sha256:
        raise ValueError(f"builderUtils.cpp SHA-256 mismatch: {original['source']}")
    members = subprocess.check_output(["ar", "t", str(original_archive)], text=True).splitlines()
    if members.count("builderUtils.cpp.o") != 1:
        raise ValueError("SDK archive does not contain exactly one builderUtils.cpp.o")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "parksight_isolated_edgellm_builder_v1",
        "status": "running",
        "sdk_revision": actual_revision,
        "original_sha256": original,
        "patch_sha256": sha256_file(patch),
    }
    try:
        staging = output / "sdk"
        cloned = staging / "cpp/builder/builderUtils.cpp"
        cloned.parent.mkdir(parents=True)
        # The migrated source mixes CRLF and LF; normalize only the isolated copy.
        cloned.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        shutil.copy2(edge_root / "cpp/builder/builderUtils.h", cloned.parent)
        shutil.copy2(patch, output / "builder.patch")
        patch_command = ["patch", "--batch", "--forward", "-p1", "-d", str(staging), "-i", str(output / "builder.patch")]
        _run(["patch", "--dry-run", *patch_command[1:]], cwd=output, log=output / "patch_check.log")
        _run(patch_command, cwd=output, log=output / "patch_apply.log")
        if b"EDGELLM_TIMING_CACHE_PATH" not in cloned.read_bytes():
            raise RuntimeError("patch did not modify the isolated builder source")
        object_file = output / "builderUtils.cpp.o"
        flags = _flags(edge_root / "build/cpp/CMakeFiles/edgellmBuilder.dir/flags.make")
        compile_command = [
            "/usr/bin/aarch64-linux-gnu-g++", *flags,
            "-c", str(cloned), "-o", str(object_file),
        ]
        _run(compile_command, cwd=staging, log=output / "compile.log")
        archive = output / "libedgellmBuilder.a"
        shutil.copy2(original_archive, archive)
        _run(["ar", "rcs", str(archive), str(object_file)], cwd=output, log=output / "archive.log")
        if subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines().count("builderUtils.cpp.o") != 1:
            raise RuntimeError("isolated archive did not replace builderUtils.cpp.o")
        link_dir = edge_root / "build/examples/llm"
        link_command = shlex.split((link_dir / "CMakeFiles/llm_build.dir/link.txt").read_text())
        link_command[link_command.index("-o") + 1] = str(output / "llm_build")
        link_command = [str(archive) if arg == "../../cpp/libedgellmBuilder.a" else arg for arg in link_command]
        if str(archive) not in link_command:
            raise ValueError("SDK link command did not reference expected archive")
        _run(link_command, cwd=link_dir, log=output / "link.log")
        builder = output / "llm_build"
        if b"EDGELLM_TIMING_CACHE_PATH" not in builder.read_bytes() or b"--profilingDetailed" not in builder.read_bytes():
            raise RuntimeError("isolated executable lacks requested controls")
        result.update(
            status="succeeded",
            isolated_sha256={
                "source": sha256_file(cloned),
                "object": sha256_file(object_file),
                "archive": sha256_file(archive),
                "builder": sha256_file(builder),
            },
            commands={"compile": compile_command, "link": link_command},
        )
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["original_unchanged"] = original == {
            "source": sha256_file(source),
            "archive": sha256_file(original_archive),
            "builder": sha256_file(original_builder),
        }
        (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    result = prepare(
        edge_root=args.edge_llm_root,
        patch=args.patch,
        output=args.output,
        expected_revision=args.expected_revision,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "commands"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

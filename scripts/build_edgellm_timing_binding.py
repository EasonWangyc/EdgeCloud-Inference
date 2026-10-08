"""Build an isolated timing binding using existing CMake flags and native libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    working = root / "build/experimental/pybind"
    target = working / "CMakeFiles/_edgellm_runtime.dir"
    definitions = {}
    for line in (target / "flags.make").read_text().splitlines():
        key, sep, value = line.partition(" = ")
        if sep and key in {"CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"}:
            definitions[key] = shlex.split(value)
    if len(definitions) != 3:
        raise ValueError("missing CMake CXX flags")
    link = shlex.split((target / "link.txt").read_text())
    old_object = "CMakeFiles/_edgellm_runtime.dir/edgellm_pybind.cpp.o"
    if link.count(old_object) != 1 or link.count("-o") != 1:
        raise ValueError("unsupported CMake binding link command")
    output_index = link.index("-o") + 1
    module_name = Path(link[output_index]).name
    if not module_name.startswith("_edgellm_runtime.") or not module_name.endswith(".so"):
        raise ValueError("unsupported binding output name")
    link[output_index] = str(output / module_name)
    link[link.index(old_object)] = str(output / "edgellm_pybind.cpp.o")
    compile_command = [link[0], *definitions["CXX_DEFINES"], *definitions["CXX_INCLUDES"],
                       *definitions["CXX_FLAGS"], "-c", str(output / "source/experimental/pybind/edgellm_pybind.cpp"),
                       "-o", str(output / "edgellm_pybind.cpp.o")]
    # All other file operands remain existing read-only linker inputs.
    inputs = [target / "flags.make", target / "link.txt"]
    original_module = root / "build/pybind" / module_name
    if original_module.is_file():
        inputs.append(original_module)
    for arg in link[1:]:
        if not arg.startswith("-") and arg.endswith((".a", ".so", ".o")):
            path = Path(arg)
            path = path if path.is_absolute() else working / path
            if path.resolve().is_relative_to(output):
                continue
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append(path.resolve())
    return {"working_directory": str(working), "compile_command": compile_command,
            "link_command": link, "module": str(output / module_name),
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in dict.fromkeys(inputs)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-record", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    record = json.loads(args.patch_record.read_text())
    source = args.edge_llm_root.resolve() / "experimental/pybind/edgellm_pybind.cpp"
    if sha256(source) != record["source_sha256"] or sha256(args.patch) != record["patch_sha256"]:
        parser.error("source or patch identity changed since preparation")
    plan = build_plan(args.edge_llm_root, output)
    if not args.execute:
        print(json.dumps({"status": "ready_for_explicit_execution", **plan}, indent=2))
        return 0
    output.mkdir(parents=True)
    copied = output / "source/experimental/pybind/edgellm_pybind.cpp"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    copied_patch = output / "timing.patch"
    copied_patch.write_bytes(args.patch.read_bytes())
    result = {"status": "running", "source_before_sha256": record["source_sha256"],
              "build_script_sha256": sha256(Path(__file__)),
              "patch_sha256": record["patch_sha256"], "plan": plan,
              "evidence_boundary": "Binding-only build; existing native libraries reused; no GPU inference"}
    destination = output / "build_record.json"
    def save():
        destination.write_text(json.dumps(result, indent=2) + "\n")
    started = time.perf_counter()
    save()
    try:
        with (output / "build.log").open("x") as log:
            for command in (["git", "apply", "--check", str(copied_patch)],
                            ["git", "apply", str(copied_patch)]):
                subprocess.run(command, cwd=output / "source", stdout=log, stderr=log, check=True)
            if sha256(copied) != record["resulting_source_sha256"]:
                raise ValueError("applied binding source does not match prepared identity")
            for command in (plan["compile_command"], plan["link_command"]):
                subprocess.run(command, cwd=plan["working_directory"], stdout=log, stderr=log, check=True)
        changed_inputs = [entry["path"] for entry in plan["inputs"]
                          if sha256(Path(entry["path"])) != entry["sha256"]]
        if sha256(source) != record["source_sha256"] or changed_inputs:
            raise ValueError(f"original source or link inputs changed during build: {changed_inputs}")
        module = Path(plan["module"])
        result.update(status="built", module_sha256=sha256(module), module_size_bytes=module.stat().st_size,
                      source_after_sha256=sha256(copied), original_inputs_unchanged=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        save()
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

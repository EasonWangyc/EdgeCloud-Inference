"""Build a source-bound Core overlay without changing native archives."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

try:
    from scripts.build_edgellm_timing_binding import build_plan, sha256
except ModuleNotFoundError:
    from build_edgellm_timing_binding import build_plan, sha256


def overlay_plan(root: Path, output: Path, binding_object: Path, expected_sha: str,
                 *, source_relative_path: str = "cpp/common/trtUtils.cpp",
                 core_archive: Path | None = None, core_archive_sha256: str | None = None) -> dict:
    if source_relative_path not in {"cpp/common/trtUtils.cpp", "cpp/runtime/llmInferenceRuntime.cpp"}:
        raise ValueError("unsupported overlay source")
    member_name = Path(source_relative_path).name + ".o"
    root, output, binding_object = root.resolve(), output.resolve(), binding_object.resolve()
    if sha256(binding_object) != expected_sha:
        raise ValueError("timing binding object SHA-256 changed")
    plan = build_plan(root, output)
    flags_path = root / "build/cpp/CMakeFiles/edgellmCore.dir/flags.make"
    definitions = {}
    for line in flags_path.read_text().splitlines():
        key, sep, value = line.partition(" = ")
        if sep and key in {"CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"}:
            definitions[key] = shlex.split(value)
    if len(definitions) != 3:
        raise ValueError("missing Core CMake CXX flags")
    link = plan["link_command"]
    slot = link.index(str(output / "edgellm_pybind.cpp.o"))
    link[slot] = str(binding_object)
    archives = [arg for arg in link if Path(arg).name == "libedgellmCore.a"]
    if len(archives) != 1:
        raise ValueError("expected exactly one Core archive")
    original_archive = (Path(plan["working_directory"]) / archives[0]).resolve()
    if core_archive is not None:
        if sha256(core_archive) != core_archive_sha256:
            raise ValueError("candidate Core archive SHA-256 changed")
        original_archive = core_archive.resolve()
        plan["inputs"].append({"path": str(original_archive), "sha256": sha256(original_archive)})
    archive = output / "libcandidateCore.a"
    link[link.index(archives[0])] = str(archive)
    archive_link = root / "build/cpp/CMakeFiles/edgellmCore.dir/link.txt"
    archiver = shlex.split(archive_link.read_text().splitlines()[0])[0]
    members = subprocess.run([archiver, "t", str(original_archive)], check=True,
                             capture_output=True, text=True).stdout.splitlines()
    if members.count(member_name) != 1:
        raise ValueError("Core archive must contain one overlay target object")
    plan.update(original_archive=str(original_archive), candidate_archive=str(archive),
                archiver=archiver, archive_members=members,
                member_name=member_name,
                archive_command=[archiver, "r", str(archive), str(output / member_name)])
    link.append("-Wl,-Map=" + str(output / "link.map"))
    plan["compile_command"] = [link[0], *definitions["CXX_DEFINES"],
                               *definitions["CXX_INCLUDES"], "-I" + str((root / source_relative_path).parent),
                               *definitions["CXX_FLAGS"], "-c", str(output / "source" / source_relative_path),
                               "-o", str(output / member_name)]
    for path in (binding_object, flags_path, archive_link, *sorted((root / "cpp").rglob("*.h"))):
        plan["inputs"].append({"path": str(path), "sha256": sha256(path)})
    return plan


def verify_link_map(text: str, member_name: str = "trtUtils.cpp.o") -> None:
    member = re.escape(member_name)
    if not re.search(r"libcandidateCore\.a\([^)]*" + member + r"\)", text):
        raise ValueError("link map does not contain the candidate overlay object")
    if re.search(r"libedgellmCore\.a\([^)]*" + member + r"\)", text):
        raise ValueError("original archive overlay object was also selected")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", type=Path, required=True)
    parser.add_argument("--binding-object", type=Path, required=True)
    parser.add_argument("--binding-object-sha256", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-record", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--core-archive", type=Path)
    parser.add_argument("--core-archive-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.edge_llm_root.resolve(), args.output_directory.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("output must be a fresh directory outside the native source checkout")
    record = json.loads(args.patch_record.read_text())
    if record["source_relative_path"] not in {"cpp/common/trtUtils.cpp", "cpp/runtime/llmInferenceRuntime.cpp"}:
        parser.error("unexpected patch target")
    if (args.core_archive is None) != (args.core_archive_sha256 is None):
        parser.error("Core archive path and SHA-256 must be supplied together")
    source = root / record["source_relative_path"]
    if sha256(source) != record["source_sha256"] or sha256(args.patch) != record["patch_sha256"]:
        parser.error("source or patch identity changed since preparation")
    plan = overlay_plan(root, output, args.binding_object, args.binding_object_sha256,
                        source_relative_path=record["source_relative_path"],
                        core_archive=args.core_archive, core_archive_sha256=args.core_archive_sha256)
    if not args.execute:
        print(json.dumps({"status": "ready_for_explicit_execution", **plan}, indent=2))
        return 0
    output.mkdir(parents=True)
    copied = output / "source" / record["source_relative_path"]
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    patch = output / "reader.patch"
    patch.write_bytes(args.patch.read_bytes())
    result = {"status": "running", "plan": plan, "patch_record": record,
              "script_sha256": sha256(Path(__file__)), "original_source_sha256": sha256(source),
              "evidence_boundary": "One Core object overlay plus existing timing binding; no GPU inference"}
    destination = output / "build_record.json"
    started = time.perf_counter()
    try:
        with (output / "build.log").open("x") as log:
            for command in (["git", "apply", "--check", str(patch)], ["git", "apply", str(patch)]):
                subprocess.run(command, cwd=output / "source", stdout=log, stderr=log, check=True)
            if sha256(copied) != record["resulting_source_sha256"]:
                raise ValueError("patched source identity mismatch")
            shutil.copyfile(plan["original_archive"], plan["candidate_archive"])
            for command in (plan["compile_command"], plan["archive_command"], plan["link_command"]):
                subprocess.run(command, cwd=plan["working_directory"], stdout=log, stderr=log, check=True)
        verify_link_map((output / "link.map").read_text(), plan["member_name"])
        members = subprocess.run([plan["archiver"], "t", plan["candidate_archive"]], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
        if members != plan["archive_members"]:
            raise ValueError("candidate archive members changed")
        member = subprocess.run([plan["archiver"], "p", plan["candidate_archive"], plan["member_name"]],
                                check=True, capture_output=True).stdout
        if member != (output / plan["member_name"]).read_bytes():
            raise ValueError("candidate archive contains a different overlay object")
        changed = [entry["path"] for entry in plan["inputs"]
                   if sha256(Path(entry["path"])) != entry["sha256"]]
        if sha256(source) != record["source_sha256"] or changed:
            raise ValueError(f"original source or native inputs changed: {changed}")
        module = Path(plan["module"])
        result.update(status="built", module_sha256=sha256(module), module_size_bytes=module.stat().st_size,
                      link_map_sha256=sha256(output / "link.map"), original_inputs_unchanged=True,
                      candidate_archive_sha256=sha256(Path(plan["candidate_archive"])),
                      original_archive_object_selected=False)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

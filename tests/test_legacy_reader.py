"""Engine-reader file semantics and source-bound isolated patch application."""

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.build_edgellm_legacy_reader import overlay_plan, verify_link_map
from scripts.prepare_edgellm_legacy_reader import LEGACY_READER, reader_patch


SOURCE = '''#include "trtUtils.h"
#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
// modern reader
#endif
void deserialize()
{
#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
// modern deserialization
#else
    file_io::MmapReader mmapReader(enginePath);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime.deserializeCudaEngine(mmapReader.getData(), mmapReader.getSize()));
#endif
}
'''


class LegacyReaderTests(unittest.TestCase):
    def test_patch_applies_with_exact_lf_crlf_and_mixed_source_identity(self):
        for newline in ("\n", "\r\n", "mixed"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as directory:
                text = SOURCE.replace("\n", "\r\n" if newline == "mixed" else newline)
                if newline == "mixed":
                    start, end = text.index("#else"), text.index("#endif", text.index("#else"))
                    text = text[:start] + text[start:end].replace("\r\n", "\n") + text[end:]
                source = text.encode() + b"// preserved mixed line\n"
                patch, expected = reader_patch(source, hashlib.sha256(source).hexdigest())
                root = Path(directory)
                target = root / "cpp/common/trtUtils.cpp"
                target.parent.mkdir(parents=True)
                target.write_bytes(source)
                patch_file = root / "reader.patch"
                patch_file.write_bytes(patch.encode())
                subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=root, check=True)
                subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True)
                result = target.read_bytes()
                self.assertEqual(hashlib.sha256(result).hexdigest(), expected)
                self.assertTrue(result.endswith(b"// preserved mixed line\n"))

    def test_rejects_drift_and_unknown_layout(self):
        source = SOURCE.encode()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            reader_patch(source, "0" * 64)
        for changed in (source.replace(b"mmapReader.getSize()", b"other()"), source + b"FileStreamReaderLegacy"):
            with self.assertRaises(ValueError):
                reader_patch(changed, hashlib.sha256(changed).hexdigest())

    def test_link_map_rejects_archive_duplicate_and_missing_overlay(self):
        verify_link_map("/candidate/libcandidateCore.a(trtUtils.cpp.o)\n")
        for text in ("LOAD other.cpp.o", "LOAD candidate/trtUtils.cpp.o",
                     "libcandidateCore.a(trtUtils.cpp.o)\nlibedgellmCore.a(trtUtils.cpp.o)"):
            with self.assertRaises(ValueError):
                verify_link_map(text)

    def test_build_rejects_changed_binding_object_before_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj = root / "timing.o"
            obj.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "binding object SHA-256"):
                overlay_plan(root, root / "output", obj, "0" * 64)

    @unittest.skipUnless(shutil.which("ar"), "archive plan test requires a host archiver")
    def test_plan_redirects_whole_archive_link_to_copy_and_keeps_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "build/experimental/pybind/CMakeFiles/_edgellm_runtime.dir"
            target.mkdir(parents=True)
            core = root / "build/cpp/CMakeFiles/edgellmCore.dir"
            core.mkdir(parents=True)
            flags = 'CXX_DEFINES = -DTEST\nCXX_INCLUDES = -I/include\nCXX_FLAGS = -fPIC\n'
            (target / "flags.make").write_text(flags)
            (core / "flags.make").write_text(flags)
            (core / "link.txt").write_text(shutil.which("ar") + " qc libedgellmCore.a\n")
            archive = root / "build/cpp/libedgellmCore.a"
            member = root / "trtUtils.cpp.o"
            member.write_bytes(b"original object")
            subprocess.run([shutil.which("ar"), "qc", str(archive), str(member)], check=True)
            before = archive.read_bytes()
            (target / "link.txt").write_text(
                '/usr/bin/g++ -shared -o ../../pybind/_edgellm_runtime.test.so '
                'CMakeFiles/_edgellm_runtime.dir/edgellm_pybind.cpp.o '
                '-Wl,--whole-archive ../../cpp/libedgellmCore.a -Wl,--no-whole-archive\n')
            binding = root / "timing.o"
            binding.write_bytes(b"verified binding")
            output = root / "candidate"
            plan = overlay_plan(root, output, binding, hashlib.sha256(binding.read_bytes()).hexdigest())
            self.assertFalse(output.exists())
            self.assertIn(str(output / "libcandidateCore.a"), plan["link_command"])
            self.assertNotIn("../../cpp/libedgellmCore.a", plan["link_command"])
            self.assertIn("-Wl,--whole-archive", plan["link_command"])
            self.assertEqual(plan["archive_members"], ["trtUtils.cpp.o"])
            self.assertEqual(archive.read_bytes(), before)

    @unittest.skipUnless(shutil.which("g++"), "reader I/O test requires a host C++ compiler")
    def test_host_reader_handles_multichunk_eof_invalid_requests_and_preserves_input(self):
        prefix = '''#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 3
namespace nvinfer1 { class IStreamReader { public: virtual ~IStreamReader() = default;
virtual int64_t read(void*, int64_t) noexcept = 0; }; }
'''
        main = '''int main(int argc, char** argv) {
    if (argc != 4) return 1;
    FileStreamReaderLegacy reader(argv[1]);
    if (reader.read(nullptr, 0) != 0 || reader.read(nullptr, 1) != -1) return 2;
    std::vector<std::byte> bytes(5 * 1024 * 1024 + 17);
    if (reader.read(bytes.data(), -1) != -1) return 3;
    if (reader.read(bytes.data(), bytes.size() + 100) != static_cast<int64_t>(bytes.size())) return 4;
    for (size_t i = 0; i < bytes.size(); ++i)
        if (bytes[i] != static_cast<std::byte>(i % 251)) return 5;
    if (reader.read(bytes.data(), 1) != 0) return 6;
    for (int i = 2; i < 4; ++i) {
        bool rejected = false;
        try { FileStreamReaderLegacy invalid(argv[i]); } catch (std::runtime_error const&) { rejected = true; }
        if (!rejected) return 7;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.cpp"
            source.write_text(prefix + LEGACY_READER + main)
            image = root / "engine.bin"
            image.write_bytes((bytes(range(251)) * 22000)[:5 * 1024 * 1024 + 17])
            before = hashlib.sha256(image.read_bytes()).hexdigest()
            empty = root / "empty.bin"
            empty.touch()
            executable = root / "reader-test"
            subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
                            "-o", str(executable)], check=True, capture_output=True)
            subprocess.run([str(executable), str(image), str(empty), str(root / "missing")], check=True)
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before)

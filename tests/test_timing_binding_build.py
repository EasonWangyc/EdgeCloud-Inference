"""Isolated outputs and unchanged CMake link operands for binding-only builds."""

import tempfile
import unittest
from pathlib import Path

from scripts.build_edgellm_timing_binding import build_plan


class TimingBindingBuildTests(unittest.TestCase):
    def test_plan_redirects_only_binding_object_and_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            working = root / "build/experimental/pybind"
            target = working / "CMakeFiles/_edgellm_runtime.dir"
            target.mkdir(parents=True)
            flags = target / "flags.make"
            flags.write_text('CXX_DEFINES = -DTEST\nCXX_INCLUDES = -I"/include path"\nCXX_FLAGS = -fPIC -std=c++17\n')
            device = target / "cmake_device_link.o"
            device.write_bytes(b"original device link")
            library = root / "build/cpp/libedgellmCore.a"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"original archive")
            original_module = root / "build/pybind/_edgellm_runtime.test.so"
            original_module.parent.mkdir()
            original_module.write_bytes(b"original binding")
            link = target / "link.txt"
            link.write_text('/usr/bin/g++ -shared -o ../../pybind/_edgellm_runtime.test.so '
                            'CMakeFiles/_edgellm_runtime.dir/edgellm_pybind.cpp.o '
                            'CMakeFiles/_edgellm_runtime.dir/cmake_device_link.o ../../cpp/libedgellmCore.a\n')
            output = root / "candidate"
            plan = build_plan(root, output)
            self.assertFalse(output.exists())
            self.assertIn(str(output / "edgellm_pybind.cpp.o"), plan["link_command"])
            self.assertIn(str(output / "_edgellm_runtime.test.so"), plan["link_command"])
            self.assertIn("CMakeFiles/_edgellm_runtime.dir/cmake_device_link.o", plan["link_command"])
            self.assertIn("../../cpp/libedgellmCore.a", plan["link_command"])
            self.assertIn("-I/include path", plan["compile_command"])
            self.assertEqual({row["path"] for row in plan["inputs"]},
                             {str(flags), str(link), str(device), str(library), str(original_module)})
            self.assertEqual(device.read_bytes(), b"original device link")
            self.assertEqual(library.read_bytes(), b"original archive")

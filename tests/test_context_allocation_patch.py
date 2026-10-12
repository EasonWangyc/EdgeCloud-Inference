"""Default/deferred allocation order and source-bound patch application."""

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_edgellm_context_allocation import context_allocation_patch


SOURCE = '''#include <string>
void initialize(bool hasDraft)
{
    int64_t const baseContextMemorySize = 16;
    int64_t const strategyContextMemorySize = 0;
    int64_t sharedContextMemorySize = std::max(baseContextMemorySize, strategyContextMemorySize);
    mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kUINT8, "LLMInferenceRuntime::mSharedExecContextMemory");
    mBaseExecutor->setContextMemory(mSharedExecContextMemory);
    if (mDecoderRegistry)
    {
        mDecoderRegistry->setContextMemory(mSharedExecContextMemory);
    }
    LOG_INFO(
        "Preallocated base execution context memory: %zu bytes (base requires: %zu, strategy requires: %zu)",
        static_cast<size_t>(sharedContextMemorySize), static_cast<size_t>(baseContextMemorySize),
        static_cast<size_t>(strategyContextMemorySize));
    events.push_back("visual");
    int64_t const requiredSharedContextMemorySize = std::max(baseContextMemorySize, visionSize);
    if (requiredSharedContextMemorySize > sharedContextMemorySize)
    {
        sharedContextMemorySize = requiredSharedContextMemorySize;
        mSharedExecContextMemory = rt::Tensor({sharedContextMemorySize}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "LLMInferenceRuntime::mSharedExecContextMemory");
        mBaseExecutor->setContextMemory(mSharedExecContextMemory);
        if (mDecoderRegistry)
        {
            mDecoderRegistry->setContextMemory(mSharedExecContextMemory);
        }
    }
}
'''


class ContextAllocationTests(unittest.TestCase):
    def test_patch_application_preserves_lf_and_crlf(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as directory:
                source = SOURCE.replace("\n", newline).encode()
                patch, expected = context_allocation_patch(source, hashlib.sha256(source).hexdigest())
                root = Path(directory)
                target = root / "cpp/runtime/llmInferenceRuntime.cpp"
                target.parent.mkdir(parents=True)
                target.write_bytes(source)
                patch_file = root / "context.patch"
                patch_file.write_bytes(patch.encode())
                subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=root, check=True)
                subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True)
                updated = target.read_bytes()
                self.assertEqual(hashlib.sha256(updated).hexdigest(), expected)
                if newline == "\r\n":
                    self.assertNotIn(b"\n", updated.replace(b"\r\n", b""))

    def test_rejects_drift_duplicate_switch_and_layout_changes(self):
        source = SOURCE.encode()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            context_allocation_patch(source, "0" * 64)
        for changed in (source + b"EDGELLM_DEFER_CONTEXT_ALLOCATION",
                        source.replace(b"requiredSharedContextMemorySize >", b"unknown >")):
            with self.assertRaises(ValueError):
                context_allocation_patch(changed, hashlib.sha256(changed).hexdigest())

    @unittest.skipUnless(shutil.which("g++"), "allocation order test requires a host C++ compiler")
    def test_default_deferred_and_growth_order_rejects_invalid_and_draft(self):
        prefix = '''#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <string>
std::vector<std::string> events;
int64_t visionSize = 8;
namespace nvinfer1 { enum class DataType { kUINT8 }; }
namespace rt { enum class DeviceType { kGPU }; struct Tensor {
Tensor() = default;
Tensor(std::initializer_list<int64_t>, DeviceType, nvinfer1::DataType, char const*) { events.push_back("allocate"); }
}; }
struct Executor { void setContextMemory(rt::Tensor&) {} } executor;
Executor* mBaseExecutor = &executor;
Executor* mDecoderRegistry = nullptr;
rt::Tensor mSharedExecContextMemory;
#define LOG_INFO(...) do {} while (false)
'''
        main = '''int main() {
    for (char const* option : {"unset", "0", "1"}) {
        if (std::string(option) == "unset") unsetenv("EDGELLM_DEFER_CONTEXT_ALLOCATION");
        else setenv("EDGELLM_DEFER_CONTEXT_ALLOCATION", option, 1);
        events.clear(); initialize(false);
        std::vector<std::string> expected = std::string(option) == "1"
            ? std::vector<std::string>{"visual", "allocate"}
            : std::vector<std::string>{"allocate", "visual"};
        if (events != expected) return 1;
    }
    visionSize = 32;
    events.clear(); initialize(false);
    if (events != std::vector<std::string>{"visual", "allocate"}) return 2;
    setenv("EDGELLM_DEFER_CONTEXT_ALLOCATION", "0", 1);
    events.clear(); initialize(false);
    if (events != std::vector<std::string>{"allocate", "visual", "allocate"}) return 3;
    for (char const* option : {"invalid", "1"}) {
        setenv("EDGELLM_DEFER_CONTEXT_ALLOCATION", option, 1);
        bool rejected = false;
        try { initialize(true); } catch (std::invalid_argument const&) { rejected = true; }
        if (!rejected) return 4;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "cpp/runtime/llmInferenceRuntime.cpp"
            target.parent.mkdir(parents=True)
            target.write_text(SOURCE)
            patch, _ = context_allocation_patch(SOURCE.encode(), hashlib.sha256(SOURCE.encode()).hexdigest())
            patch_file = root / "context.patch"
            patch_file.write_bytes(patch.encode())
            subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True)
            program = root / "order.cpp"
            program.write_text(prefix + target.read_text() + main)
            executable = root / "order"
            compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(program),
                                       "-o", str(executable)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(executable)], check=True)

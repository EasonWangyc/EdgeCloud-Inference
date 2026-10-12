"""Prepare a source-bound opt-in TensorRT 10.x host stream reader experiment."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


LEGACY_READER = '''#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR < 7
class FileStreamReaderLegacy : public nvinfer1::IStreamReader
{
public:
    explicit FileStreamReaderLegacy(std::filesystem::path const& path)
    {
        mFd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (mFd < 0)
        {
            throw std::runtime_error("Cannot open engine stream: " + path.string());
        }
        struct stat status{};
        if (fstat(mFd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size <= 0)
        {
            close(mFd);
            mFd = -1;
            throw std::runtime_error("Engine stream must be a nonempty regular file: " + path.string());
        }
        mSize = status.st_size;
    }

    FileStreamReaderLegacy(FileStreamReaderLegacy const&) = delete;
    FileStreamReaderLegacy& operator=(FileStreamReaderLegacy const&) = delete;

    ~FileStreamReaderLegacy() override
    {
        if (mFd >= 0)
        {
            close(mFd);
        }
    }

    int64_t read(void* destination, int64_t nbBytes) noexcept override
    {
        if (nbBytes < 0 || (nbBytes > 0 && destination == nullptr))
        {
            return -1;
        }
        int64_t const remaining = std::min(nbBytes, mSize - mOffset);
        int64_t completed = 0;
        while (completed < remaining)
        {
            // Copy straight into TensorRT's host buffer. Bound file-cache pressure
            // per chunk; DONTNEED is advisory and does not alter the engine file.
            size_t const chunk = static_cast<size_t>(std::min<int64_t>(remaining - completed, 4 * 1024 * 1024));
            ssize_t const count = pread(mFd, static_cast<std::byte*>(destination) + completed, chunk, mOffset);
            if (count < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return -1;
            }
            if (count == 0)
            {
                break;
            }
            static_cast<void>(posix_fadvise(mFd, mOffset, count, POSIX_FADV_DONTNEED));
            mOffset += count;
            completed += count;
        }
        return completed;
    }

private:
    int mFd{-1};
    int64_t mSize{0};
    int64_t mOffset{0};
};
#endif

'''


def reader_patch(source: bytes, expected_sha256: str) -> tuple[str, str]:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("deserialization source SHA-256 changed")
    text = source.decode()
    if "FileStreamReaderLegacy" in text:
        raise ValueError("legacy stream reader already exists")
    newline = "\r\n" if "\r\n" in text else "\n"
    anchor = '#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)'
    fallback_template = '''    file_io::MmapReader mmapReader(enginePath);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime.deserializeCudaEngine(mmapReader.getData(), mmapReader.getSize()));'''
    candidates = [fallback_template.replace("\n", ending) for ending in ("\n", "\r\n")]
    matches = [candidate for candidate in candidates if text.count(candidate) == 1]
    if len(matches) != 1:
        raise ValueError("unsupported TensorRT deserialization layout")
    fallback = matches[0]
    fallback_newline = "\r\n" if "\r\n" in fallback else "\n"
    if text.count(anchor) != 2 or text.count(fallback) != 1 or text.count('#include "trtUtils.h"') != 1:
        raise ValueError("unsupported TensorRT deserialization layout")
    updated = text.replace('#include "trtUtils.h"', '#include "trtUtils.h"' + newline + '#include <cstdlib>', 1)
    updated = updated.replace(anchor, LEGACY_READER.replace("\n", newline) + anchor, 1)
    replacement = '''    char const* readerOption = std::getenv("EDGELLM_LEGACY_STREAM_READER");
    if (readerOption != nullptr && std::string_view(readerOption) != "0" && std::string_view(readerOption) != "1")
    {
        throw std::invalid_argument("EDGELLM_LEGACY_STREAM_READER must be 0 or 1");
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    if (readerOption != nullptr && std::string_view(readerOption) == "1")
    {
#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR < 7
        FileStreamReaderLegacy reader(enginePath);
        LOG_INFO("Using TensorRT legacy host stream reader for %s", enginePath.c_str());
        engine.reset(runtime.deserializeCudaEngine(reader));
#else
        throw std::invalid_argument("Legacy host stream reader requires TensorRT 10.0 through 10.6");
#endif
    }
    else
    {
        file_io::MmapReader mmapReader(enginePath);
        engine.reset(runtime.deserializeCudaEngine(mmapReader.getData(), mmapReader.getSize()));
    }'''.replace("\n", fallback_newline)
    updated = updated.replace(fallback, replacement, 1)
    patch = "".join(difflib.unified_diff(text.splitlines(keepends=True), updated.splitlines(keepends=True),
                                      fromfile="a/cpp/common/trtUtils.cpp", tofile="b/cpp/common/trtUtils.cpp"))
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
    patch, result_sha = reader_patch(args.source.read_bytes(), args.source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as file:
        file.write(patch)
    with metadata.open("x") as file:
        json.dump({"source_sha256": args.source_sha256, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                   "resulting_source_sha256": result_sha, "source_relative_path": "cpp/common/trtUtils.cpp",
                   "switch": "EDGELLM_LEGACY_STREAM_READER=1", "status": "prepared",
                   "scope": "Opt-in TensorRT 10.x before 10.7; no checkout edits, compilation or GPU proof"}, file, indent=2)
        file.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

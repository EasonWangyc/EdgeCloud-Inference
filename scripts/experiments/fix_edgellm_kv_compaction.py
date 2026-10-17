"""Select KV eviction copy strides by engine layout in Edge-LLM v0.9.1.

The dense engine needs its original [batch, K/V, head, token, dim] copy path;
the paged engine uses an NHD [K/V, batch, token, head, dim] pool. Applying
one set of strides to both engines corrupts surviving requests on eviction.
Run this against the fixed SDK checkout before rebuilding the native runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCE = Path("cpp/kernels/speculative/batchEvictKernels.cu")
HEADER = Path("cpp/kernels/speculative/batchEvictKernels.h")
MANAGER = Path("cpp/runtime/hybridCacheManager.cpp")
OLD = """    int64_t const headStride = static_cast<int64_t>(maxSeqLen) * HEAD_DIM;
    int64_t const kvStride = static_cast<int64_t>(numKVHeads) * headStride;
    int64_t const batchStride = 2 * kvStride;
"""
NEW = """    // The owning pool is [K/V, batch, token, head, dim]. Keep both halves
    // separate and copy one head across the live token prefix of each row.
    int64_t const tokenStride = static_cast<int64_t>(numKVHeads) * HEAD_DIM;
    int64_t const batchStride = static_cast<int64_t>(maxSeqLen) * tokenStride;
    int64_t const kvStride = static_cast<int64_t>(maxBatchSize) * batchStride;
"""
OLD_COPY = """        int32_t const elemsPerKV = seqLen * HEAD_DIM;
        int64_t const srcOffset = oldBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;
        int64_t const dstOffset = newBatchIdx * batchStride + kvIdx * kvStride + kvHeadIdx * headStride;

        T const* srcPtr = kvCache + srcOffset;
        T* dstPtr = kvCache + dstOffset;

        int32_t const numVecs = elemsPerKV / VEC_SIZE;
        for (int32_t vecIdx = threadIdx.x; vecIdx < numVecs; vecIdx += threadsPerBlock)
        {
            Vec vec;
            vec.load(srcPtr + vecIdx * VEC_SIZE);
            vec.store(dstPtr + vecIdx * VEC_SIZE);
        }
"""
NEW_COPY = """        int64_t const srcOffset = kvIdx * kvStride + oldBatchIdx * batchStride + kvHeadIdx * HEAD_DIM;
        int64_t const dstOffset = kvIdx * kvStride + newBatchIdx * batchStride + kvHeadIdx * HEAD_DIM;

        T const* srcPtr = kvCache + srcOffset;
        T* dstPtr = kvCache + dstOffset;

        constexpr int32_t kVectorsPerHead = HEAD_DIM / VEC_SIZE;
        int32_t const numVecs = seqLen * kVectorsPerHead;
        for (int32_t vecIdx = threadIdx.x; vecIdx < numVecs; vecIdx += threadsPerBlock)
        {
            int32_t const tokenIdx = vecIdx / kVectorsPerHead;
            int32_t const headVecIdx = vecIdx % kVectorsPerHead;
            int64_t const elementOffset = static_cast<int64_t>(tokenIdx) * tokenStride + headVecIdx * VEC_SIZE;
            Vec vec;
            vec.load(srcPtr + elementOffset);
            vec.store(dstPtr + elementOffset);
        }
"""


PAGED_STRIDES = """    // Paged engine pool: [K/V, batch, token, head, dim].
    int64_t const pagedTokenStride = static_cast<int64_t>(numKVHeads) * HEAD_DIM;
    int64_t const pagedBatchStride = static_cast<int64_t>(maxSeqLen) * pagedTokenStride;
    int64_t const pagedKVStride = static_cast<int64_t>(maxBatchSize) * pagedBatchStride;
"""
FINAL_STRIDES = OLD + "\n" + PAGED_STRIDES


def _indented(value: str) -> str:
    return "".join("    " + line if line.strip() else line for line in value.splitlines(keepends=True))


PAGED_COPY = (NEW_COPY.replace("kvStride", "pagedKVStride")
              .replace("batchStride", "pagedBatchStride")
              .replace("tokenStride", "pagedTokenStride"))
FINAL_COPY = ("        if (pagedKV)\n        {\n" + _indented(PAGED_COPY)
              + "        }\n        else\n        {\n" + _indented(OLD_COPY)
              + "        }\n")


def _replace(data: bytes, old: str, new: str, *, expected: int = 1) -> bytes:
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    old_bytes = old.encode().replace(b"\n", newline)
    new_bytes = new.encode().replace(b"\n", newline)
    if data.count(old_bytes) != expected:
        raise ValueError(f"expected {expected} matches for {old[:80]!r}; found {data.count(old_bytes)}")
    return data.replace(old_bytes, new_bytes)


def patch_sdk(sdk_root: Path, *, apply: bool) -> str:
    source, header, manager = (sdk_root / path for path in (SOURCE, HEADER, MANAGER))
    originals = {path: path.read_bytes() for path in (source, header, manager)}
    changed = dict(originals)
    source_text = changed[source].replace(b"\r\n", b"\n")
    if FINAL_COPY.encode() in source_text and FINAL_STRIDES.encode() in source_text:
        if (b"bool pagedKV" in source_text
                and b"bool pagedKV" in changed[header]
                and b"mConfig.kvConfig.numPages > 0" in changed[manager]):
            return "already patched"
        raise ValueError("partial layout-selective patch found")
    if OLD_COPY.encode() in source_text and OLD.encode() in source_text:
        changed[source] = _replace(changed[source], OLD, FINAL_STRIDES)
        changed[source] = _replace(changed[source], OLD_COPY, FINAL_COPY)
    elif NEW_COPY.encode() in source_text and NEW.encode() in source_text:
        # Upgrade the earlier paged-only experiment without touching its backup.
        changed[source] = _replace(changed[source], NEW, FINAL_STRIDES)
        changed[source] = _replace(changed[source], NEW_COPY, FINAL_COPY)
    else:
        raise ValueError("unknown KV compact kernel layout")

    changed[source] = _replace(changed[source],
        "    int32_t oldActiveBatch)\n{\n    static_assert(HEAD_DIM",
        "    int32_t oldActiveBatch, bool pagedKV)\n{\n    static_assert(HEAD_DIM")
    changed[source] = _replace(changed[source],
        "    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, cudaStream_t stream)\n{",
        "    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, bool pagedKV,\n    cudaStream_t stream)\n{")
    changed[source] = _replace(changed[source],
        "                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch);",
        "                layerInfos, batchMappingPtr, srcKVLengthsPtr, maxBatchSize, oldActiveBatch, pagedKV);",
        expected=4)
    changed[header] = _replace(changed[header],
        "    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, cudaStream_t stream);",
        "    int32_t maxKVHeads, int32_t maxBatchSize, int32_t oldActiveBatch, int32_t newActiveBatch, bool pagedKV,\n    cudaStream_t stream);")
    changed[manager] = _replace(changed[manager],
        "            mConfig.kvConfig.kvCacheType, group.maxKVHeads, mConfig.kvConfig.maxBatchSize, oldBatch, newBatch, stream);",
        "            mConfig.kvConfig.kvCacheType, group.maxKVHeads, mConfig.kvConfig.maxBatchSize, oldBatch, newBatch,\n            mConfig.kvConfig.numPages > 0, stream);")
    if apply:
        for path, data in changed.items():
            if data != originals[path]:
                path.write_bytes(data)
        return "patched"
    return "patch available"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(patch_sdk(args.sdk_root, apply=args.apply))


if __name__ == "__main__":
    main()

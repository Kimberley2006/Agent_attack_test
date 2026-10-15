"""检查本地分片是否与 safetensors 索引和各分片头声明的大小一致。"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import MODEL_REVISION  # noqa: E402


def check_model_weights(model_dir: Path) -> tuple[list[str], int, int]:
    problems = []
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return [f"缺少索引: {index_path}"], 0, 0
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    actual_total = 0
    expected_total = 0
    for name in shards:
        path = model_dir / name
        if not path.is_file():
            problems.append(f"缺少分片: {name}")
            continue
        actual = path.stat().st_size
        actual_total += actual
        try:
            with path.open("rb") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                if header_size > 16 * 1024 * 1024:
                    raise ValueError(f"异常头长度 {header_size}")
                header = json.loads(handle.read(header_size))
            tensors = [value for key, value in header.items()
                       if key != "__metadata__"]
            if not tensors:
                raise ValueError("分片头没有张量")
            expected = 8 + header_size + max(
                tensor["data_offsets"][1] for tensor in tensors)
            expected_total += expected
            if actual != expected:
                problems.append(f"{name}: 实际 {actual:,} B，头声明 {expected:,} B")
        except (OSError, ValueError, KeyError, json.JSONDecodeError,
                struct.error) as exc:
            problems.append(f"{name}: 无法解析分片头: {exc}")

    tree_dir = model_dir / ".cache/huggingface/trees"
    if tree_dir.is_dir() and not (tree_dir / f"{MODEL_REVISION}.json").is_file():
        problems.append(f"本地 Hugging Face 元数据不含冻结 revision {MODEL_REVISION}")
    return problems, actual_total, expected_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    problems, actual, expected = check_model_weights(args.model_dir)
    print(f"分片文件：实际 {actual:,} B；按分片头应为 {expected:,} B")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    print("模型分片大小、索引和本地 revision 元数据检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

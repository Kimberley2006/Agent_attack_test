"""按冻结 SHA-256 恢复损坏的 WebShop JSON；下载完成后才替换原文件。"""
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ce990fff5aee388db2706f07820c578ab68e0453"


def digest(path):
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def restore(path, expected):
    if digest(path) == expected:
        print(f"verified {path.name}", flush=True)
        return
    partial = path.with_suffix(".json.download")
    if not partial.exists() and path.exists():
        shutil.copyfile(path, partial)
    url = f"https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/{REVISION}/{path.name}"
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("下载续传偏移不一致")
                print(f"download {path.name}: offset={offset if append else 0}, status={response.status}", flush=True)
                last = time.monotonic()
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        if time.monotonic() - last >= 30:
                            print(f"downloaded {output.tell():,} bytes", flush=True)
                            last = time.monotonic()
            if digest(partial) != expected:
                # 有可能原文件不是正确前缀；下一次从零重新下载。
                partial.rename(partial.with_name(partial.name + f".mismatch-{time.time_ns()}"))
                raise RuntimeError("下载文件 SHA-256 不匹配，保留诊断副本并重新下载")
            if path.exists():
                path.rename(path.with_name(path.name + f".corrupt-{time.time_ns()}"))
            partial.replace(path)
            print(f"restored and verified {path.name}", flush=True)
            return
        except Exception as exc:
            print(f"attempt {attempt + 1}/6: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(10)


if __name__ == "__main__":
    fingerprints = json.loads((ROOT / "task_registry.json").read_text())["data_fingerprint"]
    for name in ("items_shuffle", "items_ins_v2", "items_human_ins"):
        restore(ROOT / "webshop_repo/data" / f"{name}.json", fingerprints[f"{name}_sha256"])

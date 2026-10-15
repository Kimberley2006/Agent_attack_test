"""Load and snapshot the frozen experiment configuration."""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, data: dict, path: Path):
        self._data = data
        self.path = path

    def __getattr__(self, name):
        try:
            v = self._data[name]
        except KeyError:
            raise AttributeError(name)
        return Config(v, self.path) if isinstance(v, dict) else v

    def get(self, dotted, default=None):
        cur = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)


def load_config(path=None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "baseline_v1.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data, path)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_code(run_dir: Path, config: Config) -> dict:
    """Copy config + hash every project source file into the run dir (freeze)."""
    snap = {
        "config": config.as_dict(),
        "config_file": str(config.path),
        "config_sha256": file_sha256(config.path),
        "sources": {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pkg_dir = PROJECT_ROOT / "agent_attack"
    for f in sorted(list(pkg_dir.glob("*.py")) + list((PROJECT_ROOT / "scripts").glob("*.py"))):
        snap["sources"][str(f.relative_to(PROJECT_ROOT))] = file_sha256(f)
    (run_dir / "config_snapshot.json").write_text(json.dumps(snap, indent=2))
    return snap


def make_run_dir(config: Config, tag: str = "") -> Path:
    """Create a timestamped run directory; big artifacts live under /cache."""
    root = Path(config.get("run.output_root"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{config.get('run.run_id_prefix')}-{tag}-{stamp}" if tag else \
        f"{config.get('run.run_id_prefix')}-{stamp}"
    run_dir = root / run_id
    (run_dir / "tensors").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_model_call_id(task_id, agent_step) -> str:
    return f"t{task_id}_s{agent_step}"

#!/usr/bin/env bash
# 从本机 main conda 环境克隆 WebShop 实验环境，并补齐分支运行器依赖。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONDA_BIN=${AA_CONDA_BIN:-/data/wangxiang/conda/bin/conda}
SOURCE_ENV=${AA_SOURCE_ENV:-/data/wangxiang/conda/envs/main}
ENV_PREFIX=${AA_WEBSHOP_ENV:-$PWD/.conda-envs/webshop}
PYTHON="$ENV_PREFIX/bin/python"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PWD/.pip-cache}

if [ ! -x "$PYTHON" ]; then
  "$CONDA_BIN" create --prefix "$ENV_PREFIX" --clone "$SOURCE_ENV" --offline -y
fi

pip_args=(--no-input --disable-pip-version-check --timeout 30 --retries 2)
if [ -n "${AA_WHEELHOUSE:-}" ]; then
  pip_args+=(--no-index --find-links "$AA_WHEELHOUSE")
fi
"$PYTHON" -m pip install "${pip_args[@]}" -r configs/e0_webshop_pip.txt

if [ -n "${AA_SPACY_MODEL_WHEEL:-}" ]; then
  "$PYTHON" -m pip install "${pip_args[@]}" "$AA_SPACY_MODEL_WHEEL"
elif [ -n "${AA_WHEELHOUSE:-}" ]; then
  shopt -s nullglob
  model_wheels=("$AA_WHEELHOUSE"/en_core_web_sm-*.whl)
  shopt -u nullglob
  if [ "${#model_wheels[@]}" -ne 1 ]; then
    echo "离线目录中需要恰好一个 en_core_web_sm-*.whl；或设置 AA_SPACY_MODEL_WHEEL" >&2
    exit 2
  fi
  "$PYTHON" -m pip install "${pip_args[@]}" "${model_wheels[0]}"
else
  "$PYTHON" -m spacy download en_core_web_sm
fi

"$PYTHON" -m pip check
"$PYTHON" - <<'PY'
import spacy
from agent.env_wrapper import ensure_webshop_importable
spacy.load("en_core_web_sm")
ensure_webshop_importable()
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
print("E0 WebShop conda 环境导入检查通过")
PY

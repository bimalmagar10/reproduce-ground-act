#!/usr/bin/env bash
# One-time environment setup on the H100 node (login/data-transfer node is fine for the downloads).
# Creates one Python 3.12 venv holding BOTH alpamayo1_5 (Alpamayo 1.5) and alpamayo_r1 (Alpamayo-R1):
# the two repos pin identical torch==2.8.0 / transformers==4.57.1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
mkdir -p "$GROUNDACT_WORK" && cd "$GROUNDACT_WORK"

[ -d alpamayo1.5 ] || git clone https://github.com/NVlabs/alpamayo1.5.git
[ -d alpamayo ]    || git clone https://github.com/NVlabs/alpamayo.git
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

cd alpamayo1.5
uv venv "$GROUNDACT_WORK/venv" --python 3.12
# shellcheck disable=SC1091
source "$GROUNDACT_WORK/venv/bin/activate"
if command -v nvcc >/dev/null; then
  uv sync --active
else
  echo "nvcc not found: installing without flash-attn (run eval with --attn sdpa)"
  uv sync --active --no-install-package flash-attn
fi
uv pip install --python "$(which python)" -e "$GROUNDACT_WORK/alpamayo" --no-deps
uv pip install --python "$(which python)" -e "$HERE" --no-deps
uv pip install --python "$(which python)" "physical_ai_av>=0.2.3" pytest

# Configs / processors the two checkpoints resolve at load time (weights are local already).
hf download Qwen/Qwen3-VL-2B-Instruct --exclude "*.safetensors" "*.bin" "*.pth" >/dev/null   # processor (both)
hf download Qwen/Qwen3-VL-8B-Instruct --exclude "*.safetensors" "*.bin" "*.pth" >/dev/null   # Alpamayo-R1 VLM config
hf download nvidia/Cosmos-Reason2-8B --exclude "*.safetensors" "*.bin" "*.pth" >/dev/null     # Alpamayo 1.5 VLM config

python - <<'PY'
import torch, alpamayo1_5, alpamayo_r1, physical_ai_av, groundact_repro
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("packages ok")
PY
echo "done. next: source $HERE/env.sh && source $GROUNDACT_WORK/venv/bin/activate && python $HERE/scripts/smoke_test.py --model \$ALPAMAYO_15"

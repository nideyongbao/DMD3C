#!/usr/bin/env bash
# DMD3C inference reproduction setup, run from the workspace root.
# Tested on Ubuntu 22.04 / 24.04 with NVIDIA driver >= 535 and CUDA toolkit 12.x.
#
# Usage (from this directory):
#   bash setup_rtx4060.sh
#
# Optional environment variables:
#   DMD3C_REPO        DMD3C upstream URL  (default: https://github.com/Sharpiless/DMD3C.git)
#   BPNET_REPO        BP-Net upstream URL (default: https://github.com/kakaxi314/BP-Net.git)
#   DMD3C_DIR         local DMD3C clone path  (default: ./DMD3C)
#   BPNET_DIR         local BP-Net clone path (default: ./BP-Net)
#   KITTI_DRIVE       KITTI raw drive number under 2011_09_26 (default: 0048)
#   TORCH_CUDA_ARCH   nvcc target SM (default: 8.9 for RTX 4060 / Ada)
#   PYTHON_VERSION    Python version for uv venv (default: 3.9)
#   SKIP_DATA=1       skip KITTI raw download
#   SKIP_BUILD=1      skip BpOps compilation
#   http_proxy / https_proxy honored when exported beforehand.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMD3C_REPO="${DMD3C_REPO:-https://github.com/Sharpiless/DMD3C.git}"
BPNET_REPO="${BPNET_REPO:-https://github.com/kakaxi314/BP-Net.git}"
DMD3C_DIR="${DMD3C_DIR:-$ROOT/DMD3C}"
BPNET_DIR="${BPNET_DIR:-$ROOT/BP-Net}"
KITTI_DRIVE="${KITTI_DRIVE:-0048}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.9}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"

CKPT_URL="https://github.com/Sharpiless/DMD3C/releases/download/pretrain-checkpoints/dmd3c_distillation_depth_anything_v2.pth"
KITTI_CALIB_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"
KITTI_DRIVE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_${KITTI_DRIVE}/2011_09_26_drive_${KITTI_DRIVE}_sync.zip"

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[err]\033[0m   %s\n' "$*" >&2; exit 1; }

###############################################################################
# 1. Pre-flight checks
###############################################################################
log "Workspace root: $ROOT"
log "DMD3C dir:      $DMD3C_DIR"
log "BP-Net dir:     $BPNET_DIR"

command -v git    >/dev/null || die "git not found. sudo apt install -y git"
command -v wget   >/dev/null || die "wget not found. sudo apt install -y wget"
command -v gcc    >/dev/null || die "gcc not found. sudo apt install -y build-essential"
command -v rsync  >/dev/null || die "rsync not found. sudo apt install -y rsync"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. install NVIDIA driver first."
if ! command -v nvcc >/dev/null; then
  warn "nvcc not on PATH. If BpOps build fails, install CUDA toolkit (>=12.1) and re-run."
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "GPU detected: $GPU_NAME"

###############################################################################
# 2. uv install (idempotent: skip if already present, otherwise install + PATH)
###############################################################################
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found, installing from https://astral.sh/uv/install.sh"
  command -v curl >/dev/null || die "curl not found. sudo apt install -y curl"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installer drops binary in ~/.local/bin or ~/.cargo/bin
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -x "$d/uv" ] && export PATH="$d:$PATH"
  done
  command -v uv >/dev/null 2>&1 || die "uv install failed; check $HOME/.local/bin/uv"
  warn "uv was just installed. After this script finishes, add to your shell rc:"
  warn '    export PATH="$HOME/.local/bin:$PATH"'
fi
log "uv version: $(uv --version)"

###############################################################################
# 3. Clone DMD3C and BP-Net
###############################################################################
if [ ! -d "$DMD3C_DIR/.git" ]; then
  log "Cloning DMD3C from $DMD3C_REPO"
  git clone --depth 1 "$DMD3C_REPO" "$DMD3C_DIR"
else
  log "DMD3C already cloned, skipping"
fi

if [ ! -d "$BPNET_DIR/.git" ]; then
  log "Cloning BP-Net from $BPNET_REPO"
  git clone --depth 1 "$BPNET_REPO" "$BPNET_DIR"
else
  log "BP-Net already cloned, skipping"
fi

log "Overlaying DMD3C files onto BP-Net"
rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'outputs' \
  --exclude 'checkpoints/*.pth' \
  --exclude 'datas/kitti/raw' \
  "$DMD3C_DIR/" "$BPNET_DIR/"

###############################################################################
# 4. In-place patches (in case the upstream DMD3C clone predates these fixes)
###############################################################################
log "Applying inference patches if missing"

# Patch utils_infer.py: accept chpt as a file path or directory
if grep -q "save_path = os.path.join('checkpoints', self.cfg.chpt)" "$BPNET_DIR/utils_infer.py"; then
  python - "$BPNET_DIR/utils_infer.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = (
    "        if 'chpt' in self.cfg:\n"
    "            self.ddp_log(f'resume CHECKPOINTS')\n"
    "            save_path = os.path.join('checkpoints', self.cfg.chpt)\n"
    "            cp = torch.load(os.path.join(save_path, 'result_ema.pth'), map_location=torch.device('cpu'))\n"
)
new = (
    "        if 'chpt' in self.cfg:\n"
    "            self.ddp_log(f'resume CHECKPOINTS')\n"
    "            chpt_path = self.cfg.chpt\n"
    "            if os.path.isdir(chpt_path):\n"
    "                chpt_path = os.path.join(chpt_path, 'result_ema.pth')\n"
    "            cp = torch.load(chpt_path, map_location=torch.device('cpu'))\n"
)
if old in src:
    p.write_text(src.replace(old, new))
    print(f"patched {p}")
else:
    print(f"no patchable block found in {p}; please verify manually")
PY
fi

# Patch demo.sh default GPU id
sed -i 's|gpus=\[2\]|gpus=[0]|' "$BPNET_DIR/demo.sh" 2>/dev/null || true

###############################################################################
# 5. Python venv via uv (created at workspace root, shared between host and BP-Net)
###############################################################################
VENV_DIR="$ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating uv venv at $VENV_DIR with Python $PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV="$VENV_DIR"

log "Installing PyTorch 2.3.1 (cu121) + runtime deps"
uv pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1
uv pip install \
  hydra-core==1.3.2 omegaconf einops timm \
  opencv-python open3d imutils tqdm tensorboard \
  matplotlib pillow h5py

###############################################################################
# 6. Build BpOps CUDA extension into the root venv
###############################################################################
cd "$BPNET_DIR"
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  log "Compiling BpOps for sm_${TORCH_CUDA_ARCH/./} into $VENV_DIR"
  pushd exts >/dev/null
    rm -rf build BpOps.egg-info
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH" \
      CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
      python setup.py install
  popd >/dev/null
  python -c "import torch, BpOps; print('BpOps loaded OK:', BpOps.__file__)"
else
  log "SKIP_BUILD=1, skipping BpOps compilation"
fi

###############################################################################
# 7. Download checkpoint
###############################################################################
mkdir -p checkpoints outputs
if [ ! -s checkpoints/dmd3c_kitti.pth ]; then
  log "Downloading DMD3C checkpoint (~344 MB)"
  wget -nv -O checkpoints/dmd3c_kitti.pth "$CKPT_URL"
else
  log "Checkpoint already present, skipping download"
fi

###############################################################################
# 8. Download KITTI raw demo data
###############################################################################
if [ "${SKIP_DATA:-0}" != "1" ]; then
  KITTI_RAW="datas/kitti/raw"
  mkdir -p "$KITTI_RAW"
  pushd "$KITTI_RAW" >/dev/null
    if [ ! -f "2011_09_26/calib_cam_to_cam.txt" ]; then
      log "Downloading KITTI calib (~4 KB)"
      wget -nv -O 2011_09_26_calib.zip "$KITTI_CALIB_URL"
      python -c "import zipfile; zipfile.ZipFile('2011_09_26_calib.zip').extractall('.')"
    fi
    DRIVE_DIR="2011_09_26/2011_09_26_drive_${KITTI_DRIVE}_sync"
    if [ ! -d "$DRIVE_DIR/image_02/data" ]; then
      log "Downloading KITTI drive_${KITTI_DRIVE}_sync"
      wget -nv -O "drive_${KITTI_DRIVE}.zip" "$KITTI_DRIVE_URL"
      python -c "import zipfile; zipfile.ZipFile('drive_${KITTI_DRIVE}.zip').extractall('.')"
    fi
  popd >/dev/null

  if [ "$KITTI_DRIVE" != "0048" ]; then
    log "Patching demo.py base sequence to drive_${KITTI_DRIVE}"
    sed -i "s|2011_09_26_drive_0048_sync|2011_09_26_drive_${KITTI_DRIVE}_sync|" demo.py
  fi
else
  log "SKIP_DATA=1, skipping KITTI raw download"
fi

###############################################################################
# 9. Done
###############################################################################
cat <<EOF

================================================================================
 Setup finished. The shared uv venv lives at:

   $VENV_DIR

 To run inference:

   source $VENV_DIR/bin/activate
   cd $BPNET_DIR
   bash demo.sh

 Outputs (88 PNGs for drive_${KITTI_DRIVE}, 4 per frame) will be written to:

   $BPNET_DIR/outputs/

 Disk footprint:
   - PyTorch + deps   ~3.5 GB ($VENV_DIR)
   - DMD3C checkpoint  344 MB
   - KITTI drive_${KITTI_DRIVE} ~80 MB (22 frames)
================================================================================
EOF

# DMD³C 推理一键复现指南 (RTX 4060 / Ubuntu)

本指南配套 `setup_rtx4060.sh`，目标是在一台普通的 RTX 4060 (8 GB) Ubuntu 机器上从零跑通 DMD³C 推理 demo。

## 1. 硬件 / 系统要求

| 项 | 最低 | 已验证通过 |
|---|---|---|
| GPU | 任何支持 CUDA 12.1 的 NVIDIA 卡，**显存 ≥ 4 GB** | RTX 4060 8 GB / RTX 3060 12 GB / H20 96 GB |
| NVIDIA Driver | ≥ 535 (CUDA 12.1 runtime 兼容) | 570.195.03 |
| CUDA toolkit | 12.1 - 12.8（用于编译 `BpOps`） | 12.8 |
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 24.04 |
| 磁盘 | ≥ 6 GB 可用空间 | — |
| 联网 | 能访问 github.com 与 KITTI S3 (avg-kitti) | — |

> RTX 4060 是 Ada 架构, SM 版本 `8.9` — 脚本默认 `TORCH_CUDA_ARCH=8.9`。其他卡按下表覆盖：
> 30 系 (Ampere) `8.6` / 40 系 (Ada) `8.9` / H100/H20 (Hopper) `9.0` / A100 `8.0`。

## 2. 一键运行

工作区根目录（即本仓 clone 后所在目录）下：

```bash
bash setup_rtx4060.sh
```

脚本执行流程（约 5–10 分钟，主要看下载速度）：

1. 检查 `git / wget / gcc / rsync / curl / nvidia-smi`
2. **如未装 uv，自动从 `astral.sh/uv/install.sh` 安装**，并把 `~/.local/bin` 加到当前 shell 的 PATH。脚本结束后建议在 `~/.bashrc`/`~/.zshrc` 永久加上 `export PATH="$HOME/.local/bin:$PATH"`。
3. `git clone` DMD3C 到 `./DMD3C/` 和 BP-Net 到 `./BP-Net/`
4. 把 DMD3C 文件覆盖到 BP-Net 工作目录中
5. 应用两处 in-place patch（`utils_infer.py` chpt 路径、`demo.sh` GPU id）
6. **在工作区根 `./` 创建 `uv venv`（路径 `./.venv`，Python 3.9）。** 此 venv 是 host 与 BP-Net 共享的——demo / train / test 都用它，不再在 BP-Net 内部建独立 venv。
7. `uv pip install` PyTorch 2.3.1+cu121 + Hydra/timm/open3d 等
8. 用 `nvcc` 编译 `BpOps` CUDA 扩展，安装到 `./.venv` 的 site-packages（针对 `TORCH_CUDA_ARCH`）
9. 下载预训练权重 `dmd3c_distillation_depth_anything_v2.pth` (~344 MB) → `BP-Net/checkpoints/dmd3c_kitti.pth`
10. 下载 KITTI raw `2011_09_26_calib.zip` + `2011_09_26_drive_0048_sync.zip` (~80 MB, 22 帧)
11. 解压到 `BP-Net/datas/kitti/raw/`，建立 `BP-Net/outputs/`

完成后按提示执行：

```bash
source .venv/bin/activate         # 激活根 venv
cd BP-Net && bash demo.sh
```

## 3. 可选环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `DMD3C_REPO` | `https://github.com/Sharpiless/DMD3C.git` | DMD3C 上游或自己 fork |
| `BPNET_REPO` | `https://github.com/kakaxi314/BP-Net.git` | BP-Net 上游 |
| `DMD3C_DIR` | `./DMD3C` | DMD3C clone 位置 |
| `BPNET_DIR` | `./BP-Net` | BP-Net clone 位置（也是工作目录） |
| `KITTI_DRIVE` | `0048` | KITTI raw drive 编号；非 `0048` 时脚本会 `sed` 修改 `demo.py` 的硬编码路径 |
| `TORCH_CUDA_ARCH` | `8.9` | nvcc 编译目标 SM；按你的卡修改 |
| `PYTHON_VERSION` | `3.9` | uv 创建虚拟环境的 Python 版本 |
| `SKIP_DATA=1` | — | 跳过 KITTI 下载（已自备数据时） |
| `SKIP_BUILD=1` | — | 跳过 BpOps 编译（仅用于调试） |
| `http_proxy / https_proxy` | — | 在调用脚本前 `export` 即可，所有 `wget / uv pip / git` 都会走代理 |

例：内网走代理 + 用 30 系卡 + 换序列：

```bash
export http_proxy=http://your.proxy:port
export https_proxy=$http_proxy
TORCH_CUDA_ARCH=8.6 KITTI_DRIVE=0009 bash setup_rtx4060.sh
```

## 4. 期望输出

成功跑完 `demo.sh` 后，`BP-Net/outputs/` 下应出现 22 帧 × 4 张图：

```
0000000000_image.png       # 输入 RGB（中心裁剪到 352×1216）
0000000000_lidar.png       # 投影到相机平面的稀疏 LiDAR depth
0000000000_image_vis.png   # LiDAR 点叠到 RGB 上的可视化
0000000000_depth.png       # ★ DMD3C 预测的稠密深度（JET colormap，近蓝远红）
...
0000000021_*.png
```

参考速度：单帧网络前向 RTX 4060 上预计 **~600 ms**（H20 上 ~317 ms）。

## 5. 常见问题

### 5.1 `BpOps.so` 找不到 `libc10.so`
该 CUDA 扩展依赖 PyTorch 运行库，必须**先 `import torch` 再 `import BpOps`**。仓库代码已遵循该顺序；自定义脚本调试要保证 `torch` import 在前。

### 5.2 nvcc 与 PyTorch CUDA 版本不一致
PyTorch 2.3.1 cu121 对应 CUDA 12.1 ABI，但与 12.2 - 12.8 的 nvcc 编译都兼容（向后兼容）。本机验证过的组合：driver 12.8 + nvcc 12.8 + torch cu121 ✅。

### 5.3 显存不够
demo 单帧 batch=1 占用 < 4 GB。如果你的卡更小，可以编辑 `BP-Net/demo.py` 把 `image_height/image_width` 缩小（注意一致裁切 lidar 与 K_cam）。

### 5.4 想换序列
- 短序列：`KITTI_DRIVE=0009 bash setup_rtx4060.sh`（脚本自动 sed 改 `demo.py`）
- 自定义路径：直接编辑 `BP-Net/demo.py:339` 的 `base = "..."`

### 5.5 训练复现？
本脚本只解决**推理**。第二阶段 KITTI 蒸馏训练参见 [CLAUDE.md §4](CLAUDE.md)。

## 6. 卸载 / 清理

```bash
rm -rf .venv BP-Net DMD3C        # 删除根 venv 和两个子目录（含权重、KITTI 数据）
```

`uv` 自身不会污染系统 Python；如需完全卸载：`rm -rf ~/.local/bin/uv ~/.local/share/uv ~/.cache/uv`。

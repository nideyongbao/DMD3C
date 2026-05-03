# CLAUDE.md — DMD³C 复现工作手册

本文件供 Claude Code 在新环境再次进入本工作区时使用。读完后应能直接开工：知道仓库结构、推理已经跑通的方式、训练需要什么数据、典型坑在哪。

---

## 0. 一分钟速览

- **工作区根** (`./`)：本仓 (origin: `nideyongbao/DMD3C`)，只放协调脚本与文档。
- **子仓 1** (`./DMD3C/`)：clone 自 `Sharpiless/DMD3C`，是论文官方代码（**只是 BP-Net 的"补丁包"**，不能独立运行）。
- **子仓 2** (`./BP-Net/`)：clone 自 `kakaxi314/BP-Net`，是模型主体仓 + 工作目录（`checkpoints` / `datas` / `outputs` 在这里）。
- **uv venv** (`./.venv/`)：根工作区共享的虚拟环境，由 `setup_rtx4060.sh` 创建；host 与 BP-Net 都用它。
- **入口**：`bash setup_rtx4060.sh`，从空状态拉起整套环境。
- **推理已复现成功** (H20)：22 帧 → `BP-Net/outputs/`，单帧 ~317 ms。
- **训练第二阶段**（KITTI metric fine-tune w/ distillation）代码完整；**第一阶段**（单视图伪标签预训练）数据 pipeline 还没开源 — README 标 TODO。

---

## 1. 工作区布局

```
workspace-root (./, origin: nideyongbao/DMD3C)
├── setup_rtx4060.sh        # ★ 一键搭建脚本（推理）
├── REPRO_RTX4060.md        # 用户文档
├── CLAUDE.md               # 本文件
├── README.md               # 入口
├── .gitignore              # 排除 DMD3C/、BP-Net/、.venv/
│
├── .venv/                  # ★ 共享 uv venv（Python 3.9）。host 与 BP-Net 都用这一个
│   └── lib/python3.9/site-packages/{torch, BpOps.so, hydra, ...}
│
├── DMD3C/                  # ★ 由 setup 脚本 clone（origin: Sharpiless/DMD3C）
│   ├── demo.py / demo.sh
│   ├── train_distill.py / run.sh
│   ├── test.py
│   ├── utils.py / utils_infer.py
│   ├── disp_loss.py / criteria.py / augs.py
│   ├── datasets/  models/  configs/  exts/
│   ├── checkpoints/  datas/                    # 空，下载产物落到 BP-Net 同名目录
│   ├── setup_rtx4060.sh / REPRO_RTX4060.md / CLAUDE.md   # 历史副本（commit 9ad8619 / 57028d9）
│   └── ...
│
└── BP-Net/                 # ★ 由 setup 脚本 clone（origin: kakaxi314/BP-Net）
    │   DMD3C 文件 rsync 覆盖到这里之后才能运行
    ├── checkpoints/dmd3c_kitti.pth
    ├── datas/kitti/raw/2011_09_26/...
    ├── outputs/<idx>_*.png
    ├── exts/build/         # BpOps 中间编译产物（最终装到 ../.venv/site-packages）
    ├── models/utils.py     # 含 BpOps CUDA 算子调用
    └── (来自 BP-Net 原仓 + DMD3C overlay 的所有源码)
```

`./DMD3C`、`./BP-Net`、`./.venv` 都被 `.gitignore` 排除，根仓只 track 顶层文件。

---

## 2. 三个执行阶段

| 阶段 | 入口 | 说明 |
|---|---|---|
| **Stage 1** 单视图蒸馏预训练 | _未开源_ | 用 Depth Anything V2 在大量 RGB 图上生成视差伪标签，训练 BP-Net 得到 `pretrained_mixed_singleview_256.pth`（作者直接发布权重） |
| **Stage 2** KITTI metric 微调 + 蒸馏 | `BP-Net/train_distill.py` | 在 KITTI Depth Completion 上用 GT depth + 单目伪 disparity 双重监督 |
| **Stage 3** 推理 / Demo | `BP-Net/demo.py` | 在 KITTI raw 序列上输出稠密深度 + 可视化 |

模型骨干: `models.Pre_MF_Post` — 6 级 image encoder + 6 级自顶向下 PMP 解码器，每级用 `BpOps` CUDA 算子做 LiDAR 邻域传播。`forward(I, DISP, S, K)` 返回 6 个尺度 dense depth；推理只取 `output[-1]`。参数量 89.87 M。

---

## 3. 推理复现（已验证）

### 3.1 一键脚本（推荐）
从工作区根运行：
```bash
bash setup_rtx4060.sh             # 自动 clone DMD3C + BP-Net、装 uv（如缺）、建 ./.venv、编 BpOps、下数据
source .venv/bin/activate          # ★ 激活根 venv（host 和 BP-Net 共用）
cd BP-Net && bash demo.sh
```

`setup_rtx4060.sh` 是幂等的：所有产物都先检查存在再决定是否重做，可反复运行。

### 3.2 手工分步（出错时定位用）
```bash
ROOT=$(pwd)

# 1) clone 两个上游仓
git clone --depth 1 https://github.com/Sharpiless/DMD3C.git  $ROOT/DMD3C
git clone --depth 1 https://github.com/kakaxi314/BP-Net.git  $ROOT/BP-Net

# 2) DMD3C overlay -> BP-Net
rsync -a --exclude .git --exclude .venv --exclude outputs \
    --exclude 'checkpoints/*.pth' --exclude 'datas/kitti/raw' \
    $ROOT/DMD3C/ $ROOT/BP-Net/

# 3) 应用两处 patch（如上游 DMD3C 已合并就 no-op）
# 3a) utils_infer.py: chpt 接受文件路径或目录
# 3b) demo.sh: gpus=[2] -> gpus=[0]
sed -i 's|gpus=\[2\]|gpus=[0]|' $ROOT/BP-Net/demo.sh

# 4) uv venv 在工作区根 (Python 3.9) — host 和 BP-Net 共用
cd $ROOT
uv venv --python 3.9 .venv && source .venv/bin/activate

# 5) 装 PyTorch cu121 + 依赖
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1
uv pip install hydra-core==1.3.2 omegaconf einops timm opencv-python open3d imutils tqdm tensorboard matplotlib pillow h5py

# 6) 编译 BpOps 进根 venv (sm_89=4060, sm_86=30系, sm_90=H100/H20, sm_80=A100)
cd $ROOT/BP-Net/exts
TORCH_CUDA_ARCH_LIST=8.9 CUDA_HOME=/usr/local/cuda python setup.py install
cd $ROOT/BP-Net

# 7) 下权重
mkdir -p checkpoints outputs
wget -O checkpoints/dmd3c_kitti.pth \
  https://github.com/Sharpiless/DMD3C/releases/download/pretrain-checkpoints/dmd3c_distillation_depth_anything_v2.pth

# 8) 下 KITTI raw demo 数据
mkdir -p datas/kitti/raw && cd datas/kitti/raw
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip
python -c "import zipfile; [zipfile.ZipFile(z).extractall('.') for z in ['2011_09_26_calib.zip','2011_09_26_drive_0048_sync.zip']]"
cd $ROOT/BP-Net

# 9) 跑
bash demo.sh
```

### 3.3 期望产物
`BP-Net/outputs/` 下 22 帧 × 4 张 PNG = 88 张：
- `<idx>_image.png` — 输入 RGB（中心裁剪到 352×1216）
- `<idx>_lidar.png` — 投影到相机平面的稀疏 LiDAR depth（约 4% 像素有值）
- `<idx>_image_vis.png` — LiDAR 点叠到 RGB 的 HSV 染色图
- `<idx>_depth.png` — **DMD3C 预测的稠密深度**（JET colormap, 近蓝远红）

模型输出米制深度，单帧典型范围 5–86 m, mean ≈ 21 m。H20 单帧前向 ~317 ms，RTX 4060 估计 ~600 ms。

### 3.4 已知坑（已 patch 进 setup 脚本）
| 文件 | bug | 修复方式 |
|---|---|---|
| `utils_infer.py:104-106` | `chpt` 路径会被 `os.path.join('checkpoints', cfg.chpt)` 错拼成 `checkpoints/checkpoints/X.pth/result_ema.pth` | setup 脚本里 in-place 改成 "若是目录则 join 'result_ema.pth'，若是文件直接 load" |
| `demo.sh:1` | 默认 `gpus=[2]`，单卡机崩 | setup 脚本 `sed` 改成 `gpus=[0]` |

> **加载顺序**：`BpOps.so` 链接 `libc10.so`，必须先 `import torch` 再 `import BpOps`。仓库代码已遵守，自定义脚本要小心。

---

## 4. 训练复现（用用户提供的数据）

### 4.1 数据要求

用户需要把以下数据放到 `BP-Net/datas/kitti/`：

```
BP-Net/datas/kitti/
├── data_depth_annotated/                    # KITTI Depth Completion 官方 GT
│   ├── train/<drive>/proj_depth/groundtruth/image_0[2|3]/*.png
│   └── val  /<drive>/proj_depth/groundtruth/image_0[2|3]/*.png
├── data_depth_velodyne/                     # 官方稀疏 LiDAR
│   ├── train/<drive>/proj_depth/velodyne_raw/image_0[2|3]/*.png
│   └── val  /...
├── val_selection_cropped/                   # 官方 1000 帧验证集（必需）
│   ├── groundtruth_depth/*.png
│   ├── velodyne_raw/*.png
│   ├── image/*.png
│   └── intrinsics/*.txt
└── raw/                                     # KITTI Raw 全量序列
    └── 2011_09_26/2011_09_26_drive_XXXX_sync/
        ├── image_02/{data,disp}/*.png       # ★ disp 是用户用 Depth Anything V2 离线生成的视差伪标签
        └── image_03/{data,disp}/*.png
```

**特别说明 — `disp/` 目录是 DMD³C 蒸馏的关键输入**（仓库未自动生成，需用户离线产出）：
- 灰度 PNG (uint8, 0~255)，与 `data/*.png` 同名同分辨率
- 由 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 对每张 KITTI raw RGB 跑前向，把输出的 relative depth 取倒数得到 relative disparity，min-max 归一到 [0,1] × 255 后保存为 uint8 PNG
- `BP-Net/datasets/kitti.py:99-100` 在 train 模式下从 `data_depth_annotated/.../*.png` 反推到对应的 `raw/.../disp/*.png`，路径必须严格对应
- 加载时 `augs.Norm` 做 `disp = disp / 255` 归一到 [0,1]，再在 `train_distill.py:28` 通过 `disp_gt = 1/(D_gt+0.5)` 把 GT depth 也转成视差与之做尺度位移不变损失

数据规模参考：
- `data_depth_annotated` ~14 GB
- `data_depth_velodyne` ~3 GB
- `val_selection_cropped` ~600 MB
- `raw/2011_09_26+...` 全部用到的序列 ~165 GB
- 用户生成的 `disp/` ~30 GB（与 `data/` 同数量级）

### 4.2 起点权重

```bash
cd BP-Net
wget -O checkpoints/pretrained_mixed_singleview_256.pth \
  https://github.com/Sharpiless/DMD3C/releases/download/pretrain-checkpoints/pretrained_mixed_singleview_256.pth
```

这是作者发布的 Stage 1 产物（仅用单视图图像预训练，KITTI val zero-shot RMSE 1.4251）。**Stage 2 必须从这里继续**，否则蒸馏损失会和未学好的 backbone 冲突。

### 4.3 启动命令

**4 卡 (官方推荐)**:
```bash
source .venv/bin/activate         # 根 venv
cd BP-Net

torchrun --nproc_per_node=4 --master_port 4321 train_distill.py \
    gpus=[0,1,2,3] num_workers=4 name=DMD3D_BP_KITTI \
    ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
    net=PMP data=KITTI \
    lr=5e-4 train_batch_size=2 test_batch_size=1 \
    sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
    nepoch=30 test_epoch=25 ++net.sbn=true
```

**单卡 (RTX 4060 8 GB 不够，至少需要 24 GB 显存)**:
```bash
torchrun --nproc_per_node=1 --master_port 4321 train_distill.py \
    gpus=[0] num_workers=4 name=DMD3D_BP_KITTI \
    ++chpt=checkpoints/pretrained_mixed_singleview_256.pth \
    net=PMP data=KITTI \
    lr=1.25e-4 train_batch_size=2 test_batch_size=1 \
    sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
    nepoch=30 test_epoch=25
```
单卡时按 linear scaling 把 lr 砍到 1/4 (5e-4 → 1.25e-4)，并去掉 `++net.sbn=true`（SyncBN 单卡无意义）。

### 4.4 训练流程要点

`train_distill.py:14-49` 主循环每个 batch：
1. `output = run.net(I, DISP, S, K)` — 6 尺度 dense depth list
2. `loss = MSMSE(output, depth_gt)` — 多尺度 MSE，权重 `[2⁻¹⁰, 2⁻⁸, 2⁻⁶, 2⁻⁴, 2⁻², 1]`
3. `disp_gt = 1/(depth_gt+0.5)`; `disp_loss = SSI(output[-1], disp_gt, mask=ones)` — 蒸馏损失
4. `(MSMSE losses + disp_loss).sum().backward()`
5. `EMA.update(net)`，每 iter 更新 OneCycle LR

每 epoch 末（或 `epoch >= test_epoch=25` 后每 1000 iter）跑一次 selval 集，按 RMSE 取 best EMA 权重保存到 `BP-Net/checkpoints/<name>/result_ema.pth`。

### 4.5 预期指标

按论文表，Stage 2 完成后 KITTI selval RMSE ≈ 0.71（与已发布 checkpoint 的 `best_metric_ema=0.7177` 一致）。中途 `tensorboard --logdir BP-Net/runs/<name>` 看 loss + RMSE 曲线。

### 4.6 评测官方权重 / 自训权重

```bash
source .venv/bin/activate && cd BP-Net

# selval（官方 1000 帧，本地有 GT）
python test.py gpus=[0] name=eval_dmd3c \
    ++chpt=checkpoints/dmd3c_kitti.pth \
    net=PMP data=KITTI data.testset.mode=selval \
    test_batch_size=1 metric=MetricALL

# test（官方 1000 帧，无 GT；++save=true 输出 16-bit PNG 提交 KITTI server）
python test.py gpus=[0] name=submit_dmd3c \
    ++chpt=checkpoints/dmd3c_kitti.pth \
    net=PMP data=KITTI data.testset.mode=test \
    test_batch_size=1 metric=RMSE ++save=true
```
注意 `test.py` 用的是 `utils.py:Trainer` 而不是 `utils_infer.py`，它直接 `torch.load(self.cfg.chpt)`，所以 `++chpt=` 后跟完整 .pth 文件路径即可（与 demo 不同）。

---

## 5. 环境元信息（本机已验证）

- **硬件**: NVIDIA H20 (96 GB, sm_90), 24 核 CPU, 122 GB RAM, /shard_data 3.5 TB
- **OS**: Ubuntu 24.04, kernel 6.8.0
- **驱动 / 工具链**: NVIDIA driver 570.195.03 (CUDA 12.8), nvcc 12.8, gcc 13.3
- **venv**: `./.venv` (Python 3.9.25, 工作区根), 由 `uv 0.11.1` 管理；host 与 BP-Net 共用
- **关键版本**: torch 2.3.1+cu121, torchvision 0.18.1+cu121, hydra-core 1.3.2, timm 1.0.26, einops 0.8+, open3d (装最新)
- **代理**（如需）: `export http_proxy=http://proxy.parametrix.cn:9999; export https_proxy=$http_proxy`

---

## 6. 上游仓库链接

- 工作区 root fork: <https://github.com/nideyongbao/DMD3C>
- DMD³C 论文官方仓: <https://github.com/Sharpiless/DMD3C> (CVPR 2025)
- BP-Net 基仓: <https://github.com/kakaxi314/BP-Net>
- Depth Anything V2: <https://github.com/DepthAnything/Depth-Anything-V2>
- KITTI Depth Completion: <http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion>
- KITTI Raw: <http://www.cvlibs.net/datasets/kitti/raw_data.php>
- 论文: <https://arxiv.org/abs/2503.16970>

---

## 7. 给后续 Claude 的工作提示

1. **进入新会话第一件事**：读 `CLAUDE.md`，确认 `./.venv` 是否存在；不存在就跑 `setup_rtx4060.sh`。
2. **改代码**：永远改 `BP-Net/` 下的文件（运行时实际依赖的副本）；如要回流到 DMD3C 上游，再 `cp` 回 `DMD3C/` 同名文件并在 DMD3C 子仓里 commit。
3. **commit 范围**：根仓只 commit 顶层脚本/文档；DMD3C 和 BP-Net 子仓的修改各自在自己 .git 里 commit。三层 git 不要混。
4. **不要 push**：所有 commit 都只在本地，除非用户明确说 push。
5. **代理**：在容易变慢的下载命令前 `export http_proxy=...`（已记入 memory）。
6. **GPU 架构**：不同机器的 `TORCH_CUDA_ARCH_LIST` 必须改 — H20=9.0、4060=8.9、30 系=8.6、A100=8.0。BpOps 编译失败十有八九是这个。

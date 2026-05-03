# DMD³C 复现工作区

这是一个 **工作区根仓**（origin: `nideyongbao/DMD3C`），用于在新机器上从零跑通 [DMD³C (CVPR 2025)](https://arxiv.org/abs/2503.16970) 的推理 demo，并提供训练复现说明。

仓内只保留协调脚本与文档，真正的源码会由 setup 脚本运行时 clone 到本地子目录：

| 子目录 | 来源 | 角色 |
|---|---|---|
| `DMD3C/` | <https://github.com/Sharpiless/DMD3C> | 论文官方代码（BP-Net 的补丁包） |
| `BP-Net/` | <https://github.com/kakaxi314/BP-Net> | 基础模型仓 + 工作目录（含 `.venv` / 数据 / 输出） |

两者都被 `.gitignore` 排除，互不干扰。

## 快速开始 (RTX 4060 / Ubuntu)

```bash
git clone https://github.com/nideyongbao/DMD3C.git workspace
cd workspace
bash setup_rtx4060.sh
cd BP-Net
source .venv/bin/activate
bash demo.sh
```

详细说明见 [REPRO_RTX4060.md](REPRO_RTX4060.md)，以及给 Claude Code 的工作手册 [CLAUDE.md](CLAUDE.md)。

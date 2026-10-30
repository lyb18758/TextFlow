# TextFlow

[English](<README.md>) | [简体中文](<README.zh-CN.md>) | [论文](https://arxiv.org/abs/2603.24571)

论文 **Towards Training-Free Scene Text Editing** 的代码。TextFlow 结合 **Flow Manifold Steering（FMS）** 和 **Attention Boost（AttnBoost）**，实现免训练场景文本编辑，支持 FLUX-Kontext 和 LongCat 后端。

## 安装

需要 Linux、Python 3.10/3.11、NVIDIA GPU 和 Git。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

请按显卡驱动选择兼容的 CUDA 版本，并保留 [requirements.txt](<requirements.txt>) 中固定的 diffusers 版本。

## 模型与数据

另行下载模型，放到以下目录：

```text
models/FLUX.1-Kontext-dev/
models/LongCat-Image-Edit/
```

FLUX 可通过 `--model_path` 指定模型位置；LongCat 使用上述固定路径。输入数据为 JSON 列表：

```json
[
  {"image_path": "./data/images/sample.png", "text_s": "OPEN", "text_t": "CLOSED"}
]
```

`text_s` 为原文本，`text_t` 为替换文本。路径相对于仓库根目录解析；重复记录可用唯一的 `id` 区分。

## 使用

所有命令均在仓库根目录运行。

### 单 GPU

```bash
# FLUX-Kontext
CUDA_VISIBLE_DEVICES=0 python infer_flux_text.py \
  --json_path ./data/dataset.json --exp_dir ./outputs/flux \
  --scheduler overshoot --use_att --use_flowedit

# LongCat
CUDA_VISIBLE_DEVICES=0 python infer_longcat_text.py \
  --json_path ./data/dataset.json --exp_dir ./outputs/longcat \
  --scheduler overshoot --use_att --use_flowedit
```

### 多 GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_flux_text_multi_gpu.py --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_multi --scheduler overshoot --use_att --use_flowedit
```

LongCat 将入口替换为 [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>)。每张 GPU 都需容纳完整模型。

- `--use_flowedit` 开启 FMS，`--use_att` 开启 AttnBoost；去掉前者可关闭 FMS。
- 无注意力的 Euler 采样使用 `--scheduler euler --no-use_att`。
- 结果保存在 `--exp_dir` 下。同配置、同数据集可续跑；更换配置、代码或模型时请使用新目录。

### 入口一览

| 入口 | 任务 |
| --- | --- |
| [infer_flux_text.py](<infer_flux_text.py>) | FLUX 文本替换 |
| [infer_flux_text_multi_gpu.py](<infer_flux_text_multi_gpu.py>) | FLUX 多 GPU 文本替换 |
| [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>) | FLUX 多 GPU 颜色编辑（黄色） |
| [infer_longcat_text.py](<infer_longcat_text.py>) | LongCat 文本替换，可选 FMS |
| [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>) | LongCat 多 GPU 文本替换 |
| [infer_longcat_attention.py](<infer_longcat_attention.py>) | LongCat AttnBoost，不含 FMS |
| [infer_flux_baseline.py](<infer_flux_baseline.py>) | 原生 FLUX-Kontext 对照，路径在脚本内设置 |

## 测试

CPU 回归测试仅需 Python 和 Pillow：

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s textflow_flux -p test_pipeline_safety.py -v
python -B -m unittest discover -s textflow_longcat -p test_pipeline_safety.py -v
```

[CI](https://github.com/lyb18758/TextFlow/actions/workflows/tests.yml) · [更新记录](<CHANGELOG.md>)

## 许可证

[Apache-2.0](<LICENSE>)。第三方声明见 [NOTICE](<NOTICE>) 与 [diffusers 许可证](<third_party/diffusers/LICENSE>)。模型权重和数据集遵循各自条款。

# TextFlow：免训练场景文本编辑

[English](<README.md>) | [简体中文](<README.zh-CN.md>)

**代码版本：v0.1.0** · [更新记录](<CHANGELOG.md>) · [CPU 测试](https://github.com/lyb18758/TextFlow/actions/workflows/tests.yml)

论文 **Towards Training-Free Scene Text Editing** 的研究代码。

- [论文](https://arxiv.org/abs/2603.24571)
- [代码仓库](https://github.com/lyb18758/TextFlow)

TextFlow 无需针对任务训练，即可编辑自然图像中的文本。框架结合用于保留风格和结构的 **Flow Manifold Steering（FMS）** 与用于文本渲染的 **Attention Boost（AttnBoost）**，采用注意力调制的 overshoot 采样。本快照包含 FLUX-Kontext 实现和 LongCat 适配。颜色编辑只是提示词变体，并非独立算法。

> **当前状态：**这是经过整理、清理本机路径并做了有限运行修复的研究发布快照，不是架构重构。文件按任务与职责规范命名，方法阶段与权重未改。包含纯 CPU 回归检查；依赖安装、GPU 推理及论文结果复现尚未验证。

## 1. 内容与入口

| 入口 | 用途 | Pipeline |
| --- | --- | --- |
| [infer_flux_text.py](<infer_flux_text.py>) | FLUX 单 GPU 文本替换，可选 FMS/FlowEdit 分支 | [逐步注入 pipeline](<textflow_flux/pipeline_flux_kontext_stepwise_injection.py>) |
| [infer_flux_text_multi_gpu.py](<infer_flux_text_multi_gpu.py>) | FLUX 多 GPU 文本替换 | [延迟移交 pipeline](<textflow_flux/pipeline_flux_kontext_deferred_handoff.py>) |
| [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>) | FLUX 多 GPU 颜色编辑，提示词将原文本改为黄色 | [逐步注入 pipeline](<textflow_flux/pipeline_flux_kontext_stepwise_injection.py>) |
| [infer_longcat_attention.py](<infer_longcat_attention.py>) | LongCat 单 GPU AttnBoost/overshoot 入口 | [注意力 pipeline](<textflow_longcat/pipeline_longcat_attention.py>) |
| [infer_longcat_text.py](<infer_longcat_text.py>) | LongCat 单 GPU 入口，支持实验性 FlowEdit | [FMS pipeline](<textflow_longcat/pipeline_longcat_fms.py>) |
| [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>) | LongCat 多 GPU 入口，可选实验性 FlowEdit | [FMS pipeline](<textflow_longcat/pipeline_longcat_fms.py>) |
| [infer_flux_baseline.py](<infer_flux_baseline.py>) | 原生 diffusers FLUX-Kontext 对照入口，无 CLI 解析器 | 原生 FLUX-Kontext pipeline |

入口采用 `infer_<backend>_<task>[_multi_gpu].py` 命名，启动脚本名称标明任务。后端代码分别位于 [textflow_flux](<textflow_flux/>) 与 [textflow_longcat](<textflow_longcat/>) 包。包括 AMO 在内的类名及原有 CLI 参数不因文件重命名而改变；旧实验文件名不保留 wrapper 脚本。

两个后端目录包含所需的自定义 pipeline、Transformer、注意力补丁、调度器与输出类型。本快照不包含模型权重、数据集、基准评测套件或完整的 diffusers 源码副本。

FLUX 的两种变体保留了不同的 FMS 轨迹移交策略：`deferred_handoff` 在移交步接入独立演化的引导轨迹；`stepwise_injection` 则在每个启用 FMS 的步骤将其写入采样 latent。它们描述的是保留实现的行为差异，不是严格互斥的两个阶段，也不表示质量高低或 GPU 硬件限制；两种变体不能直接互换。

## 2. 环境

建议从 **Linux、Python 3.10/3.11、NVIDIA GPU 及兼容的 CUDA 版 PyTorch** 开始。入口使用 CUDA 和 bfloat16；CPU 推理与 Windows 多 GPU 运行尚未验证。显存需求随模型与图像分辨率变化，本快照不提供已验证的最低显存值。

在仓库根目录创建干净环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Example CUDA build; select a build compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](<requirements.txt>) 保留了主要推理与模型加载依赖的原始版本，但移除了原始全环境冻结列表，包括显式固定的 NVIDIA runtime、NCCL、Triton wheel 以及未使用的绘图/OpenCV 包。这些底层与间接依赖交由相应上游包管理，因此它**不是**完整环境锁文件。

Diffusers 固定到提交 `d7a1c31f4f85bae5a9e01cdce49bd7346bd8ccd6`，原检出版本记录为 `0.37.0.dev0`。安装需要 Git 及 GitHub 访问权限。自定义模型依赖 diffusers 内部 API，未经核对不要替换版本。本整理快照尚未验证安装与依赖解析。

## 3. 模型与输入数据

请另行下载官方 FLUX.1-Kontext-dev 和/或 LongCat-Image-Edit 模型目录，并遵守各自的许可与访问条件。典型目录结构如下：

```text
TextFlow-release/
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── textflow_flux/
├── textflow_longcat/
├── models/
│   ├── FLUX.1-Kontext-dev/
│   └── LongCat-Image-Edit/
├── data/
│   ├── dataset.json
│   └── images/
│       └── sample.png
└── outputs/
```

上图中的模型、数据和输出目录由用户提供或在运行时创建，不随本快照分发。FLUX 自定义入口接受 `--model_path`；LongCat 入口直接加载上图所示的固定相对模型位置，**不提供** `--model_path` 参数。

输入 JSON 是记录列表：

```json
[
  {
    "image_path": "./data/images/sample.png",
    "text_s": "OPEN",
    "text_t": "CLOSED"
  }
]
```

`image_path` 指向源图像，`text_s` 为原文本，`text_t` 为替换文本。颜色编辑分支仍读取这一结构，但实际提示词使用源文本和固定的黄色。

**所有命令都应在仓库根目录运行。**相对路径（包括 JSON 记录中的路径）相对于进程工作目录解析，而非 JSON 所在目录。允许图像文件的主文件名重复。可选的字符串/整数 `id` 提供稳定样本标识；否则标识由图像路径与源/目标文本派生。重复标识会被拒绝，重复记录应分配不同 ID。

## 4. 用法

### FLUX：单 GPU 文本替换

```bash
CUDA_VISIBLE_DEVICES=0 python infer_flux_text.py \
  --model_path ./models/FLUX.1-Kontext-dev \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_text \
  --scheduler overshoot --c 2.0 \
  --num_inference_steps 50 --guidance_scale 2.5 \
  --use_att --use_flowedit
```

去掉 `--use_flowedit` 即可关闭 FlowEdit 增强。未使用的 `--edit_strength` 参数已移除。FlowEdit 区间参数和 `--n_avg` 仍可用；启用时须满足 `0 <= start < end <= num_inference_steps` 且 `n_avg >= 1`。不同入口的默认值不完全相同。

所有自定义入口均接受 `--use_att` / `--no-use_att`，不能传布尔字符串。注意力默认开启，唯独 LongCat 多 GPU 入口仍默认关闭。注意力要求使用 overshoot 调度器；普通 Euler 采样请用 `--scheduler euler --no-use_att`。

### FLUX：多 GPU 文本替换

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_flux_text_multi_gpu.py \
  --model_path ./models/FLUX.1-Kontext-dev \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_text_multi \
  --scheduler overshoot --use_att --use_flowedit
```

此方式在模型副本之间分配样本，不是模型切分；每张 GPU 都必须能独立容纳完整模型。

### FLUX：颜色编辑

将多 GPU 命令中的文本替换入口换成 [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>)，并使用独立输出目录。当前提示词将源文本颜色改为黄色，没有通用的 `--color` CLI 参数。保留的 [FLUX shell 启动脚本](<run_flux_color_multi_gpu.sh>) 启动的是**颜色分支**，不是多 GPU 文本替换入口。

### LongCat：单 GPU 注意力/overshoot 编辑

将模型放到上文指定的固定相对位置后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python infer_longcat_attention.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_text \
  --scheduler overshoot --c 2.0 \
  --num_inference_steps 50 --guidance_scale 2.5
```

此入口默认开启注意力。多 GPU 仅注意力编辑可使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_longcat_text_multi_gpu.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_text_multi \
  --scheduler overshoot --use_att
```

[LongCat shell 启动脚本](<run_longcat_text_multi_gpu.sh>) 同样默认仅启用注意力编辑。单 GPU 适配中启用 FMS 的命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python infer_longcat_text.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_flowedit \
  --scheduler overshoot --use_att --use_flowedit
```

LongCat 现分别编码源条件与目标条件。默认手动源描述使用 `text_s`；可选的 `--auto_src_prompt` 只从包含明确引号的替换指令中推断源描述，例如 `Please replace the word 'OPEN' with 'CLOSED'`。含糊或自定义指令需通过 pipeline API 显式提供源描述。FLUX 入口中沿用旧名的 `--auto_src_prompt` 保留自条件回退行为（以目标提示词作为源），并不执行文本提取；需要源条件 FMS 时请省略此参数。两种适配均不声称已复现论文结果。

### 现有启动配置与输出

[accelerate_config.yaml](<accelerate_config.yaml>) 和两个 shell 启动脚本保留**四 GPU** 配置（`0,1,2,3`），并非适用于所有机器的通用默认值。上面的双 GPU 示例自行指定设备和进程数。使用四 GPU 启动脚本前，请按入口声明的默认位置准备数据集，或直接使用 Python 入口并像上面的示例一样显式传入路径。

生成图像写入 `--exp_dir`，通常位于嵌套的生成图像子目录。对照入口 [infer_flux_baseline.py](<infer_flux_baseline.py>) 使用其顶部定义的固定相对输入、模型和输出路径。公共辅助逻辑位于 [inference_utils.py](<inference_utils.py>)。

运行时的清单文件与生成图像子目录名称分别为：

```text
run_config.json
generated_image/
```

- 输出名由清理后的图像主文件名与稳定样本哈希组成。每个样本的种子由基础种子与样本标识派生，不依赖索引或 rank，因此数值种子序列与旧脚本不同。
- 运行清单记录入口、实际生效的 CLI 配置和数据集摘要。续跑要求配置与数据集一致，允许改变 GPU 数量。配置不匹配，或旧 PNG 输出缺少清单时，必须使用新实验目录。清单不记录模型/代码指纹，修改任一项后都应新建目录。旧清单记录的是重命名前的入口文件名；迁移到新入口时也须新建实验目录，不应手动修改旧清单。
- PNG 以原子方式保存，跳过前会检查有效性；损坏或未完成的输出会重新运行。输出文件系统必须支持同目录原子替换及用于发布清单的硬链接。
- 多 GPU 运行先对完整数据集分片，再过滤已完成样本。空 rank 跳过模型加载，但仍参与最终同步；逐样本失败会汇总并导致非零退出。进程组超时为三小时。

## 5. 轻量回归检查

在仓库根目录运行，只需 Python 与 Pillow，无需模型，也不需要导入 torch/diffusers：

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s textflow_flux -p test_pipeline_safety.py -v
python -B -m unittest discover -s textflow_longcat -p test_pipeline_safety.py -v
```

[入口与辅助逻辑测试](<tests/test_inference_safety.py>) 覆盖 CLI 开关、稳定 ID/种子、配置保护、并发清单发布、损坏图像与原子保存处理，以及模拟的多 rank 控制流程。[FLUX 测试](<textflow_flux/test_pipeline_safety.py>) 和 [LongCat 测试](<textflow_longcat/test_pipeline_safety.py>) 通过 AST 提取与桩对象检查源条件、generator、注意力契约及输出类型。这些是回归检查，不是 GPU 集成测试。同一套检查已配置于 [CPU 测试工作流](<.github/workflows/tests.yml>)，在 Ubuntu / Python 3.10、3.11 下仅安装 Pillow，不下载模型。

## 6. 尚存限制

- 自定义 Transformer 会在运行时修改 diffusers 模块名，两个后端应使用独立进程。
- 高分辨率注意力图可能占用大量显存。文本 token 定位沿用原有假设，不是通用指令解析器。
- 稳定样本种子与 generator 传递不保证跨硬件、CUDA 内核或依赖版本的逐位一致性。
- 依赖安装、实际分布式/NCCL 运行、GPU 图像质量、耗时、显存需求、评测分数及论文复现，仍需在目标环境验证。

## 7. 许可与署名

本代码版本采用 **Apache License 2.0**，详见 [LICENSE](<LICENSE>)。保留上游原有版权声明，并在修改过的后端文件中标明 TextFlow 适配。来源与修改说明见 [NOTICE](<NOTICE>)，diffusers 许可证副本保留于 [third_party/diffusers/LICENSE](<third_party/diffusers/LICENSE>)。本仓库不分发模型权重或数据集，它们仍受各自的许可证和访问条件约束。

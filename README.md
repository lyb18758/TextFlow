# TextFlow

[English](<README.md>) | [简体中文](<README.zh-CN.md>) | [Paper](https://arxiv.org/abs/2603.24571)

Code for **Towards Training-Free Scene Text Editing**. TextFlow combines **Flow Manifold Steering (FMS)** and **Attention Boost (AttnBoost)** for training-free scene text editing, with FLUX-Kontext and LongCat backends.

## Installation

Requires Linux, Python 3.10/3.11, an NVIDIA GPU and Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Choose a CUDA build compatible with your driver. Keep the pinned diffusers version in [requirements.txt](<requirements.txt>).

## Models and data

Download the models separately into:

```text
models/FLUX.1-Kontext-dev/
models/LongCat-Image-Edit/
```

FLUX accepts `--model_path`; LongCat uses the fixed path above. Prepare a JSON list:

```json
[
  {"image_path": "./data/images/sample.png", "text_s": "OPEN", "text_t": "CLOSED"}
]
```

`text_s` is the original text and `text_t` is the replacement. Paths are relative to the repository root. Optional unique `id` fields distinguish repeated records.

## Usage

Run from the repository root.

### Single GPU

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

### Multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_flux_text_multi_gpu.py --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_multi --scheduler overshoot --use_att --use_flowedit
```

For LongCat, replace the entry with [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>). Each GPU loads a full model.

- `--use_flowedit` enables FMS; `--use_att` enables AttnBoost. Omit the former to disable FMS.
- To use Euler without attention, pass `--scheduler euler --no-use_att`.
- Results go under `--exp_dir`. Resume with the same settings and dataset; use a new directory when changing settings, code or models.

### Entry points

| Entry | Task |
| --- | --- |
| [infer_flux_text.py](<infer_flux_text.py>) | FLUX text replacement |
| [infer_flux_text_multi_gpu.py](<infer_flux_text_multi_gpu.py>) | FLUX multi-GPU text replacement |
| [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>) | FLUX multi-GPU color editing (yellow) |
| [infer_longcat_text.py](<infer_longcat_text.py>) | LongCat text replacement with optional FMS |
| [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>) | LongCat multi-GPU text replacement |
| [infer_longcat_attention.py](<infer_longcat_attention.py>) | LongCat AttnBoost without FMS |
| [infer_flux_baseline.py](<infer_flux_baseline.py>) | Native FLUX-Kontext baseline; paths set in the script |

## Tests

CPU regression tests require only Python and Pillow:

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s textflow_flux -p test_pipeline_safety.py -v
python -B -m unittest discover -s textflow_longcat -p test_pipeline_safety.py -v
```

[CI](https://github.com/lyb18758/TextFlow/actions/workflows/tests.yml) · [Changelog](<CHANGELOG.md>)

## License

[Apache-2.0](<LICENSE>). Third-party attribution: [NOTICE](<NOTICE>) and [diffusers license](<third_party/diffusers/LICENSE>). Model weights and datasets are subject to their own terms.

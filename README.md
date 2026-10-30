# TextFlow: Training-Free Scene Text Editing

[English](<README.md>) | [简体中文](<README.zh-CN.md>)

**Code release: v0.1.0** · [Changelog](<CHANGELOG.md>) · [CPU tests](https://github.com/lyb18758/TextFlow/actions/workflows/tests.yml)

Research code associated with **Towards Training-Free Scene Text Editing**.

- [Paper](https://arxiv.org/abs/2603.24571)
- [Repository](https://github.com/lyb18758/TextFlow)

TextFlow edits text in natural images without task-specific training. The framework combines **Flow Manifold Steering (FMS)** for style/structure preservation with **Attention Boost (AttnBoost)** for text rendering, using attention-modulated overshoot sampling. This snapshot contains FLUX-Kontext implementations and a LongCat adaptation. Color editing is a prompt variant, not a separate algorithm.

> **Status:** a curated, path-sanitized research release with scoped runtime fixes, not an architectural refactor. File names are normalized by task and role; method stages and weights are unchanged. CPU-only regression checks are included; installation, GPU inference and paper-result reproduction have not been validated.

## 1. Contents and entry points

| Entry | Purpose | Pipeline |
| --- | --- | --- |
| [infer_flux_text.py](<infer_flux_text.py>) | FLUX single-GPU text replacement; optional FMS/FlowEdit branch | [Stepwise-injection pipeline](<textflow_flux/pipeline_flux_kontext_stepwise_injection.py>) |
| [infer_flux_text_multi_gpu.py](<infer_flux_text_multi_gpu.py>) | FLUX multi-GPU text replacement | [Deferred-handoff pipeline](<textflow_flux/pipeline_flux_kontext_deferred_handoff.py>) |
| [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>) | FLUX multi-GPU color editing; prompt changes existing text to yellow | [Stepwise-injection pipeline](<textflow_flux/pipeline_flux_kontext_stepwise_injection.py>) |
| [infer_longcat_attention.py](<infer_longcat_attention.py>) | LongCat single-GPU AttnBoost/overshoot entry | [Attention pipeline](<textflow_longcat/pipeline_longcat_attention.py>) |
| [infer_longcat_text.py](<infer_longcat_text.py>) | LongCat single-GPU entry with experimental FlowEdit support | [FMS pipeline](<textflow_longcat/pipeline_longcat_fms.py>) |
| [infer_longcat_text_multi_gpu.py](<infer_longcat_text_multi_gpu.py>) | LongCat multi-GPU entry with optional experimental FlowEdit | [FMS pipeline](<textflow_longcat/pipeline_longcat_fms.py>) |
| [infer_flux_baseline.py](<infer_flux_baseline.py>) | Native diffusers FLUX-Kontext comparison entry; no CLI parser | Native FLUX-Kontext pipeline |

Entry names follow `infer_<backend>_<task>[_multi_gpu].py`; launcher names identify their task. Backend code lives in the `textflow_flux` and `textflow_longcat` packages. Class names and CLI options are unchanged; old experimental file names are not retained as wrapper scripts.

The backend folders contain the required custom pipelines, Transformers, attention patches, schedulers and output types. There are no model weights, datasets, benchmark evaluation suite or vendored diffusers checkout in this snapshot. The two FLUX variants retain different FMS integration schedules: deferred handoff transfers the independent steering trajectory at its handoff step, while stepwise injection writes it into the sampling latents at each active FMS step. These names describe implementation behavior, not GPU requirements or a quality ranking; the variants are not interchangeable.

## 2. Environment

Recommended starting point: **Linux, Python 3.10/3.11, an NVIDIA GPU and a compatible CUDA-enabled PyTorch build**. The entry points use CUDA and bfloat16; CPU inference and Windows multi-GPU execution are not validated. GPU memory requirements depend on the model and image resolution; this snapshot does not establish a minimum VRAM figure.

Create a clean environment and install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Example CUDA build; select a build compatible with your NVIDIA driver.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](<requirements.txt>) retains the original versions of the main inference/model-loading dependencies. It deliberately removes the original full-environment freeze, including explicit NVIDIA runtime, NCCL and Triton wheel pins and unused plotting/OpenCV packages. These low-level and transitive dependencies are left to the corresponding upstream packages. This is **not** a complete environment lockfile.

Diffusers is pinned to commit `d7a1c31f4f85bae5a9e01cdce49bd7346bd8ccd6`, recorded as `0.37.0.dev0` in the original checkout. Installation requires Git and access to GitHub. Do not replace it with a different diffusers version without checking the internal APIs used by the custom models. Installation and dependency resolution have not been exercised for this curated snapshot.

## 3. Models and input data

Download the official FLUX.1-Kontext-dev and/or LongCat-Image-Edit model directories separately, subject to their own licenses and access conditions. A typical layout is:

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

The model, data and output directories shown above are user-provided or created at runtime, not included files. FLUX custom entries accept `--model_path`. LongCat entries currently load `./models/LongCat-Image-Edit` directly and do **not** expose a `--model_path` argument.

Input JSON is a list of records:

```json
[
  {
    "image_path": "./data/images/sample.png",
    "text_s": "OPEN",
    "text_t": "CLOSED"
  }
]
```

`image_path` points to the source image, `text_s` is the original text and `text_t` is the replacement text. The color-edit branch still reads this schema, but its active prompt uses the source text and the fixed color yellow.

**Run every command from the repository root.** Relative paths, including paths inside JSON records, are resolved against the process working directory, not the JSON file location. Repeated filename stems are supported. An optional string/integer `id` provides a stable sample identity; otherwise identity is derived from the image path and source/target text. Duplicate identities are rejected; give repeated records distinct IDs.

## 4. Usage

### FLUX: single-GPU text replacement

```bash
CUDA_VISIBLE_DEVICES=0 python infer_flux_text.py \
  --model_path ./models/FLUX.1-Kontext-dev \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_text \
  --scheduler overshoot --c 2.0 \
  --num_inference_steps 50 --guidance_scale 2.5 \
  --use_att --use_flowedit
```

Omit `--use_flowedit` to disable the FlowEdit enhancement. The unused `--edit_strength` option has been removed. FlowEdit intervals and `--n_avg` remain available; when enabled, require `0 <= start < end <= num_inference_steps` and `n_avg >= 1`. Defaults differ between entries.

All custom entries accept `--use_att` / `--no-use_att` (do not pass a boolean string). Attention defaults to enabled except in the LongCat multi-GPU entry, where it remains disabled. Attention requires the overshoot scheduler; use `--scheduler euler --no-use_att` for ordinary Euler sampling.

### FLUX: multi-GPU text replacement

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_flux_text_multi_gpu.py \
  --model_path ./models/FLUX.1-Kontext-dev \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/flux_text_multi \
  --scheduler overshoot --use_att --use_flowedit
```

This distributes samples across model replicas; it is not model sharding. Each GPU must be able to hold its own model.

### FLUX: color editing

Use [infer_flux_color_multi_gpu.py](<infer_flux_color_multi_gpu.py>) in place of the text-replacement entry in the multi-GPU command, with a separate output directory. Its current prompt changes the source text color to yellow; there is no generic `--color` CLI option. The retained [FLUX shell launcher](<run_flux_color_multi_gpu.sh>) launches this **color branch**, not the multi-GPU text-replacement entry.

### LongCat: single-GPU attention/overshoot editing

Place the model at the fixed relative location described above, then run:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_longcat_attention.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_text \
  --scheduler overshoot --c 2.0 \
  --num_inference_steps 50 --guidance_scale 2.5
```

This entry enables attention by default. For multi-GPU attention-only editing:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 \
  infer_longcat_text_multi_gpu.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_text_multi \
  --scheduler overshoot --use_att
```

The [LongCat shell launcher](<run_longcat_text_multi_gpu.sh>) also defaults to attention-only editing. To enable FMS in the single-GPU adaptation:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_longcat_text.py \
  --json_path ./data/dataset.json \
  --exp_dir ./outputs/longcat_flowedit \
  --scheduler overshoot --use_att --use_flowedit
```

LongCat now encodes source and target conditions independently. The default manual source description uses `text_s`; optional `--auto_src_prompt` only infers it from an unambiguous quoted replacement instruction, such as `Please replace the word 'OPEN' with 'CLOSED'`. Ambiguous/custom instructions require an explicit source description through the pipeline API. In FLUX entries the historically named `--auto_src_prompt` retains the self-conditioning fallback (target prompt as source), not text extraction; omit it for source-conditioned FMS. Neither adaptation is claimed to reproduce paper results.

### Existing launch configuration and outputs

[accelerate_config.yaml](<accelerate_config.yaml>) and both shell launchers retain a **four-GPU** configuration (`0,1,2,3`). They are not portable defaults for every machine. The two-GPU examples above select their own devices and process count. When using the four-GPU launchers, first provide the dataset at the defaults declared by the entry, or invoke the Python entry directly with explicit paths as shown above.

Generated images are written under `--exp_dir`, usually in nested `generated_image/` directories. The baseline [infer_flux_baseline.py](<infer_flux_baseline.py>) instead uses fixed relative input/model/output paths defined at its top. Shared helpers are in [inference_utils.py](<inference_utils.py>).

- Output names combine a sanitized image stem with a stable sample hash. Each sample seed is derived from the base seed and that identity, not its index or rank; the numerical seed sequence therefore differs from older scripts.
- A `run_config.json` records the entry, effective CLI settings and dataset digest. Resume requires the same configuration and dataset; changing GPU count is allowed. A mismatch, or old PNG outputs without this manifest, requires a new experiment directory. Models/code changes are not fingerprinted: use a new directory after changing either. Earlier manifests record the previous entry filenames, so also start a new experiment directory when migrating to the renamed entries rather than editing an existing manifest.
- PNGs are saved atomically and checked before skipping. Corrupt/incomplete outputs are rerun. The output filesystem must support same-directory atomic replacement and hard links for manifest publication.
- Multi-GPU runs shard the full dataset before filtering completed samples. Empty ranks skip model loading but still participate in final synchronization; per-sample failures are aggregated and cause a nonzero exit. The process-group timeout is three hours.

## 5. Lightweight regression checks

From the repository root (Python and Pillow are sufficient; no model or torch/diffusers import is needed):

```bash
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s textflow_flux -p test_pipeline_safety.py -v
python -B -m unittest discover -s textflow_longcat -p test_pipeline_safety.py -v
```

[Entry/helper tests](<tests/test_inference_safety.py>) cover CLI switches, stable IDs/seeds, configuration guards, concurrent manifest publication, corrupt/atomic image handling and mocked multi-rank control flow. [FLUX tests](<textflow_flux/test_pipeline_safety.py>) and [LongCat tests](<textflow_longcat/test_pipeline_safety.py>) cover source-conditioning/generator/attention contracts and output types using AST extraction and stubs. These are regression checks, not GPU integration tests. The same suite is configured for Ubuntu / Python 3.10 and 3.11 in the [CPU test workflow](<.github/workflows/tests.yml>), which installs only Pillow and does not download models.

## 6. Remaining limitations

- Custom Transformers patch diffusers module names at runtime; use separate processes for the two backends.
- Attention maps can require substantial memory at higher resolutions. Text-token localization retains the original assumptions and is not general-purpose instruction parsing.
- Stable sample seeds and generator propagation do not guarantee bitwise reproducibility across hardware, CUDA kernels or dependency versions.
- Dependency installation, actual distributed execution, GPU image quality, timing, memory requirements, evaluation scores and paper reproduction still need testing in the intended environment.

## 7. Licensing and attribution

This code release is licensed under the **Apache License 2.0**; see [LICENSE](<LICENSE>). Original upstream copyright notices are retained, and modified backend files identify the TextFlow adaptations. See [NOTICE](<NOTICE>) for attribution and modification information, and [third_party/diffusers/LICENSE](<third_party/diffusers/LICENSE>) for the diffusers license copy. Model weights and datasets are not included and remain subject to their own licenses and access conditions.

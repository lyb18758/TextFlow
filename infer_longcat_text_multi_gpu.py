import argparse
import os
import torch
import math
from tqdm import tqdm
from PIL import Image
from datetime import timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from diffusers.utils import load_image
from inference_utils import atomic_save_image, is_output_complete, output_path, prepare_run, sample_seed, validate_args

from textflow_longcat.pipeline_longcat_fms import LongCatImageEditAMOPipeline
from textflow_longcat.scheduling_attention_overshoot import StochasticRFOvershotDiscreteScheduler

def run(args):
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=3))
    accelerator = Accelerator(kwargs_handlers=[init_kwargs])


    if args.scheduler == 'overshoot':
        exp_prefix = f"{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"
    else:
        exp_prefix = f"{args.scheduler}"

    file_save_dir = os.path.join(
        args.exp_dir,
        "generated_image",
        f"num_steps={str(args.num_inference_steps).zfill(4)}",
        exp_prefix
    )

    full_data = prepare_run(file_save_dir, args, __file__)
    accelerator.wait_for_everyone()

    # Shard the full dataset before skipping completed samples.
    samples_per_process = math.ceil(len(full_data) / accelerator.num_processes)
    start_idx = accelerator.process_index * samples_per_process
    assigned_data = full_data[start_idx:start_idx + samples_per_process]
    local_data = [item for item in assigned_data if not is_output_complete(output_path(file_save_dir, item))]
    print(f"GPU {accelerator.process_index}: {len(local_data)} pending samples")

    if local_data:
        print(f"GPU {accelerator.process_index}: Loading model...")
        pipe = LongCatImageEditAMOPipeline.from_pretrained(
            "./models/LongCat-Image-Edit",
            torch_dtype=torch.bfloat16
        )
        pipe.to(accelerator.device)

        if args.scheduler == 'overshoot':
            scheduler_config = pipe.scheduler.config
            scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
            overshot_func = lambda t, dt: t + dt

            pipe.scheduler = scheduler
            pipe.scheduler.set_c(args.c)
            pipe.scheduler.set_overshot_func(overshot_func)

    error_count = 0
    pbar = tqdm(local_data, disable=not accelerator.is_local_main_process)

    for i, item in enumerate(pbar):
        try:
            image_path = item['image_path']
            text_s = item['text_s']
            text_t = item['text_t']

            img_save_path = output_path(file_save_dir, item)

            if is_output_complete(img_save_path):
                continue

            img = load_image(image_path)
            width, height = img.size

            prompt = f"Please replace the word '{text_s}' with '{text_t}'"
            if accelerator.is_local_main_process:
                tqdm.write(f"GPU {accelerator.process_index} Processing: {prompt}")

            src_prompt = None
            if args.use_flowedit and not args.auto_src_prompt:
                src_prompt = f"A photo with the word '{text_s}'"

            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(sample_seed(args.seed, item))

            out = pipe(
                prompt=prompt,
                image=img,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
                use_att=args.use_att,
                src_prompt=src_prompt,
                use_flowedit_enhancement=args.use_flowedit,
                flowedit_start_step=args.flowedit_start_step,
                flowedit_end_step=args.flowedit_end_step,
                n_avg=args.n_avg,
            ).images[0]

            origin_img = out.resize((width, height), Image.LANCZOS)
            atomic_save_image(origin_img, img_save_path)

        except Exception as e:
            error_count += 1
            print(f"GPU {accelerator.process_index} Error processing {item.get('image_path', 'unknown')}: {e}")
            continue

    print(f"GPU {accelerator.process_index}: 完成所有任务。")

    total_errors = accelerator.reduce(torch.tensor(error_count, device=accelerator.device), reduction="sum").item()
    accelerator.wait_for_everyone()
    if total_errors:
        raise RuntimeError(f"Inference failed for {total_errors} samples; see per-rank errors")
    if accelerator.is_main_process:
        print(f"所有GPU推理完成，结果保存在: {file_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scheduler", type=str, default="overshoot", choices=["euler", "overshoot"])
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--json_path", type=str, default="./data/anytext-benchmark/laion_word/test1kedit.json")
    parser.add_argument("--exp_dir", type=str, default="./outputs/longcatf_anybench")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--use_att", action=argparse.BooleanOptionalAction, default=False, help="whether to use attention guidance")
    parser.add_argument("--seed", type=int, default=67280421310721)
    parser.add_argument("--use_flowedit", action="store_true")
    parser.add_argument("--flowedit_start_step", type=int, default=0)
    parser.add_argument("--flowedit_end_step", type=int, default=50)
    parser.add_argument("--n_avg", type=int, default=2)
    parser.add_argument("--auto_src_prompt", action="store_true")

    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)

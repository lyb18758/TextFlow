import argparse
import os
import torch
import math
import time
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from textflow_flux.pipeline_flux_kontext_deferred_handoff import FluxKontextEditPipelineAMO
from textflow_flux.scheduling_attention_overshoot import StochasticRFOvershotDiscreteScheduler
from diffusers.utils import load_image
from inference_utils import atomic_save_image, is_output_complete, output_path, prepare_run, sample_seed, validate_args

def run(args):
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=10800))])

    if args.scheduler == 'overshoot':
        if args.use_flowedit:
            exp_prefix = (
                f"FUSED_{args.scheduler}_c={str(args.c).zfill(4)}_"
                f"att={args.use_att}_"
                f"steps={args.flowedit_start_step}-{args.flowedit_end_step}_navg={args.n_avg}"
            )
        else:
            exp_prefix = f"KONTEXT_{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"
    else:
        exp_prefix = f"{'FUSED' if args.use_flowedit else 'KONTEXT'}_{args.scheduler}"

    file_save_dir = os.path.join(
        args.exp_dir,
        "generated_image",
        f"num_steps={str(args.num_inference_steps).zfill(4)}",
        exp_prefix
    )

    data = prepare_run(file_save_dir, args, __file__)

    if accelerator.is_main_process:

        log_file = os.path.join(file_save_dir, "experiment_config.txt")
        with open(log_file, "w", encoding="utf-8") as log_f:
            log_f.write("=" * 50 + "\n")
            log_f.write("FLUX Kontext-Edit Fusion Experiment (Multi-GPU)\n")
            log_f.write("=" * 50 + "\n\n")
            log_f.write(f"Mode: {' Kontext + FlowEdit Fusion' if args.use_flowedit else '📝 Pure Kontext'}\n")
            log_f.write(f"Model: {args.model_path}\n")
            log_f.write(f"Scheduler: {args.scheduler}\n")
            log_f.write(f"Inference Steps: {args.num_inference_steps}\n")
            log_f.write(f"Guidance Scale: {args.guidance_scale}\n")
            log_f.write(f"Use Attention: {args.use_att}\n")
            log_f.write(f"Seed: {args.seed}\n")
            log_f.write(f"Total GPUs: {accelerator.num_processes}\n\n")

            if args.use_flowedit:
                log_f.write("FlowEdit Enhancement Parameters:\n")
                log_f.write(f"  - FlowEdit Steps: [{args.flowedit_start_step}, {args.flowedit_end_step}]\n")
                log_f.write(f"  - Velocity Averaging (n_avg): {args.n_avg}\n")
                log_f.write(f"  - Auto Source Prompt: {args.auto_src_prompt}\n\n")

            log_f.write("=" * 50 + "\n\n")

    accelerator.wait_for_everyone()

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    total_samples = len(data)
    samples_per_process = math.ceil(total_samples / num_processes)
    start_idx = process_index * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_samples)

    # Shard before resume filtering so ranks never disagree about ownership.
    assigned_data = data[start_idx:end_idx]
    process_data = [item for item in assigned_data if not is_output_complete(output_path(file_save_dir, item))]
    completed_count = len(assigned_data) - len(process_data)

    print(f"\n{'='*60}")
    print(f" GPU {process_index} 启动")
    print(f" 总样本数: {total_samples}")
    print(f" 本进程处理: [{start_idx}:{end_idx}] (共{len(process_data)}个)")
    print(f" 模式: {'Kontext+FlowEdit融合' if args.use_flowedit else '纯Kontext'}")
    print(f"{'='*60}\n")

    if process_data:
        print(f"GPU {process_index} 正在加载模型...")
        start_time = time.time()

        pipe = FluxKontextEditPipelineAMO.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        pipe = pipe.to(accelerator.device)

        if args.scheduler == 'overshoot':
            scheduler_config = pipe.scheduler.config
            scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
            overshot_func = lambda t, dt: t + dt

            pipe.scheduler = scheduler
            pipe.scheduler.set_c(args.c)
            pipe.scheduler.set_overshot_func(overshot_func)

        load_time = time.time() - start_time
        print(f" GPU {process_index} 模型加载完成，耗时: {load_time:.2f}s\n")

    success_count = 0
    error_count = 0
    skip_count = completed_count

    inference_start_time = time.time()

    pbar = tqdm(
        process_data,
        desc=f"GPU {process_index}",
        position=process_index,
        leave=True
    )

    for local_idx, item in enumerate(pbar):
        try:
            image_path = item['image_path']
            text_s = item['text_s']
            text_t = item['text_t']

            if not os.path.exists(image_path):
                error_count += 1
                pbar.set_postfix({
                    'success': success_count,
                    'skip': skip_count,
                    'error': error_count,
                    'status': 'file_not_found'
                })
                continue

            img_save_path = output_path(file_save_dir, item)

            if is_output_complete(img_save_path):
                skip_count += 1
                pbar.set_postfix({
                    'success': success_count,
                    'skip': skip_count,
                    'error': error_count,
                    'status': 'skipped'
                })
                continue

            img = load_image(image_path)
            width, height = img.size

            if width <= 0 or height <= 0:
                error_count += 1
                continue

            tar_prompt = f"Please replace the word {text_s} with '{text_t}'"

            if args.use_flowedit and not args.auto_src_prompt:
                src_prompt = f"A photo with the word '{text_s}'"
            else:
                src_prompt = None

            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(sample_seed(args.seed, item))

            out = pipe(
                image=img,
                prompt=tar_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,

                use_att=args.use_att,

                use_flowedit_enhancement=args.use_flowedit,
                src_prompt=src_prompt if args.use_flowedit else None,
                flowedit_start_step=args.flowedit_start_step if args.use_flowedit else 0,
                flowedit_end_step=args.flowedit_end_step if args.use_flowedit else 24,
                n_avg=args.n_avg if args.use_flowedit else 1,

                output_type="pil",
                return_dict=True,
            ).images[0]

            origin_img = out.resize((width, height), Image.LANCZOS)
            atomic_save_image(origin_img, img_save_path)

            success_count += 1

            pbar.set_postfix({
                'success': success_count,
                'skip': skip_count,
                'error': error_count,
                'size': f'{width}x{height}'
            })

        except Exception as e:
            error_count += 1
            print(f"\n GPU {process_index} 错误 ({item.get('image_path', 'unknown')}): {e}")
            pbar.set_postfix({
                'success': success_count,
                'skip': skip_count,
                'error': error_count,
                'status': 'error'
            })
            continue

    total_inference_time = time.time() - inference_start_time

    print(f"\n{'='*60}")
    print(f" GPU {process_index} 处理完成!")
    print(f"{'='*60}")
    print(f" 统计信息:")
    print(f"  - 成功: {success_count}")
    print(f"  - 跳过: {skip_count}")
    print(f"  - 失败: {error_count}")
    print(f"  - 总耗时: {total_inference_time:.2f}s")
    if success_count > 0:
        print(f"  - 平均每张: {total_inference_time/success_count:.2f}s")
    print(f"{'='*60}\n")

    total_errors = accelerator.reduce(torch.tensor(error_count, device=accelerator.device), reduction="sum").item()
    accelerator.wait_for_everyone()
    if total_errors:
        raise RuntimeError(f"Inference failed for {total_errors} samples; see per-rank errors")

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print(" 所有GPU处理完成!")
        print(f" 结果保存在: {file_save_dir}")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" FLUX Kontext-Edit Fusion Pipeline - 多GPU推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    basic_group = parser.add_argument_group('基础参数')
    basic_group.add_argument(
        "--model_path",
        type=str,
        default="./models/FLUX.1-Kontext-dev",
        help="FLUX-Kontext模型路径"
    )
    basic_group.add_argument(
        "--json_path",
        type=str,
        default="./data/test_images_pad/dataset.json",
        help="数据集JSON文件路径"
    )
    basic_group.add_argument(
        "--exp_dir",
        type=str,
        default="./outputs/flux_kontext_fusion_multigpu",
        help="实验结果保存目录"
    )
    basic_group.add_argument(
        "--seed",
        type=int,
        default=67280421310721,
        help="随机种子基数(按稳定样本标识派生)"
    )

    scheduler_group = parser.add_argument_group('调度器参数')
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        default="overshoot",
        choices=["euler", "overshoot"],
        help="调度器类型"
    )
    scheduler_group.add_argument(
        "--c",
        type=float,
        default=2.0,
        help="Overshoot调度器的c参数"
    )
    scheduler_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )

    kontext_group = parser.add_argument_group('Kontext参数')
    kontext_group.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="引导强度(Classifier-Free Guidance)"
    )
    kontext_group.add_argument(
        "--use_att",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用注意力引导"
    )

    flowedit_group = parser.add_argument_group(' FlowEdit融合参数')
    flowedit_group.add_argument(
        "--use_flowedit",
        action="store_true",
        help="启用FlowEdit增强"
    )
    flowedit_group.add_argument(
        "--flowedit_start_step",
        type=int,
        default=0,
        help="FlowEdit开始步数"
    )
    flowedit_group.add_argument(
        "--flowedit_end_step",
        type=int,
        default=42,
        help="FlowEdit结束步数"
    )
    flowedit_group.add_argument(
        "--n_avg",
        type=int,
        default=2,
        help="速度场平均次数(1-3推荐)"
    )
    flowedit_group.add_argument(
        "--auto_src_prompt",
        action="store_true",
        help="复用目标prompt作为源条件(历史自增强模式，非文字解析)"
    )

    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)

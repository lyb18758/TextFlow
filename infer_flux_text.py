import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image

from textflow_flux.pipeline_flux_kontext_stepwise_injection import FluxKontextEditPipelineAMO
from textflow_flux.scheduling_attention_overshoot import StochasticRFOvershotDiscreteScheduler
from diffusers.utils import load_image
from inference_utils import atomic_save_image, is_output_complete, output_path, prepare_run, sample_seed, validate_args

def run(args):
    print(f" 加载 FluxKontextEdit 融合Pipeline...")
    pipe = FluxKontextEditPipelineAMO.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    if args.scheduler == 'overshoot':
        scheduler_config = pipe.scheduler.config
        scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
        overshot_func = lambda t, dt: t + dt

        if args.use_flowedit:
            exp_prefix = (
                f"FUSED_{args.scheduler}_c={str(args.c).zfill(4)}_"
                f"att={args.use_att}_"
                f"steps={args.flowedit_start_step}-{args.flowedit_end_step}_navg={args.n_avg}"
            )
        else:
            exp_prefix = f"KONTEXT_{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"

        pipe.scheduler = scheduler
        pipe.scheduler.set_c(args.c)
        pipe.scheduler.set_overshot_func(overshot_func)
    else:
        exp_prefix = f"{'FUSED' if args.use_flowedit else 'KONTEXT'}_{args.scheduler}"

    file_save_dir = os.path.join(
        args.exp_dir,
        "generated_image",
        f"num_steps={str(args.num_inference_steps).zfill(4)}",
        exp_prefix
    )
    data = prepare_run(file_save_dir, args, __file__)

    log_file = os.path.join(file_save_dir, "experiment_config.txt")
    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write("=" * 50 + "\n")
        log_f.write("FLUX Kontext-Edit Fusion Experiment\n")
        log_f.write("=" * 50 + "\n\n")
        log_f.write(f"Mode: {' Kontext + FlowEdit Fusion' if args.use_flowedit else ' Pure Kontext'}\n")
        log_f.write(f"Model: {args.model_path}\n")
        log_f.write(f"Scheduler: {args.scheduler}\n")
        log_f.write(f"Inference Steps: {args.num_inference_steps}\n")
        log_f.write(f"Guidance Scale: {args.guidance_scale}\n")
        log_f.write(f"Use Attention: {args.use_att}\n")
        log_f.write(f"Seed: {args.seed}\n\n")

        if args.use_flowedit:
            log_f.write("FlowEdit Enhancement Parameters:\n")
            log_f.write(f"  - FlowEdit Steps: [{args.flowedit_start_step}, {args.flowedit_end_step}]\n")
            log_f.write(f"  - Velocity Averaging (n_avg): {args.n_avg}\n")
            log_f.write(f"  - Auto Source Prompt: {args.auto_src_prompt}\n\n")

        log_f.write("=" * 50 + "\n\n")

    print(f" 加载数据集: {len(data)} 张图像")
    print(f" 保存目录: {file_save_dir}")
    print(f"  模式: {' Kontext + FlowEdit 融合' if args.use_flowedit else ' 纯 Kontext'}\n")

    error_count = 0
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        image_path = item['image_path']
        text_s = item['text_s']
        text_t = item['text_t']

        img_save_path = output_path(file_save_dir, item)
        if is_output_complete(img_save_path):
            continue

        img = load_image(image_path)
        width, height = img.size

        tar_prompt = f"Please replace the word '{text_s}' with '{text_t}'"

        if args.use_flowedit and not args.auto_src_prompt:
            src_prompt = f"A photo with the word '{text_s}'"

            print(f"\n[{idx+1}/{len(data)}] {image_path}")
            print(f"   FlowEdit: {text_s} → {text_t}")
            print(f"   Source: {src_prompt}")
            print(f"   Target: {tar_prompt}")
        else:
            src_prompt = None
            print(f"\n[{idx+1}/{len(data)}] {image_path}")
            print(f" Prompt: {tar_prompt}")

        generator = torch.Generator(device='cuda')
        generator.manual_seed(sample_seed(args.seed, item))

        try:
            out = pipe(
                image=img,
                prompt=tar_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,

                use_att=args.use_att,  # 注意力引导

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
            print(f" 已保存: {width}x{height}")

            with open(log_file, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{idx+1}] {image_path}: {text_s} → {text_t} ✓\n")
                log_f.write(f"[{idx+1}] {src_prompt} + {tar_prompt} \n")

        except Exception as e:
            error_count += 1
            print(f" 错误: {str(e)}")
            with open(log_file, "a", encoding="utf-8") as log_f:
                log_f.write(f"[{idx+1}] {image_path}: ERROR - {str(e)}\n")

    if error_count:
        raise RuntimeError(f"Inference failed for {error_count} samples; see errors above")

    print(f"\n{'='*60}")
    print(f" 推理完成!")
    print(f" 结果保存在: {file_save_dir}")
    print(f" 日志文件: {log_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" FLUX Kontext-Edit Fusion Pipeline - 统一推理脚本",
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
        default="./data/test_images_pad/dataset_test.json",
        help="数据集JSON文件路径"
    )
    basic_group.add_argument(
        "--exp_dir",
        type=str,
        default="./outputs/flux_kontext_flow_render",
        help="实验结果保存目录"
    )
    basic_group.add_argument(
        "--seed",
        type=int,
        default=67280421310721,
        help="随机种子"
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
        help="是否使用注意力引导(Kontext核心功能)"
    )

    flowedit_group = parser.add_argument_group(' FlowEdit融合参数')
    flowedit_group.add_argument(
        "--use_flowedit",
        action="store_true",
        help=" 启用FlowEdit增强(与Kontext深度融合)"
    )
    flowedit_group.add_argument(
        "--flowedit_start_step",
        type=int,
        default=0,
        help="FlowEdit开始步数(默认0,从第一步开始)"
    )
    flowedit_group.add_argument(
        "--flowedit_end_step",
        type=int,
        default=50, #46
        help="FlowEdit结束步数(建议≤总步数的85%)\n"
             "  越大编辑幅度越大,但可能影响细节质量"
    )
    flowedit_group.add_argument(
        "--n_avg",
        type=int,
        default=1, #2
        help="速度场平均次数(1-3推荐)\n"
             "  1: 快速模式\n"
             "  2-3: 高质量模式(更稳定但更慢)"
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

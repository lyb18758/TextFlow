import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image

from textflow_longcat.pipeline_longcat_attention import LongCatImageEditAMOPipeline
from textflow_longcat.scheduling_attention_overshoot import StochasticRFOvershotDiscreteScheduler
from diffusers.utils import load_image
from inference_utils import atomic_save_image, is_output_complete, output_path, prepare_run, sample_seed, validate_args

def run(args):
    pipe = LongCatImageEditAMOPipeline.from_pretrained(
        "./models/LongCat-Image-Edit",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    if args.scheduler == 'overshoot':
        scheduler_config = pipe.scheduler.config
        scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
        overshot_func = lambda t, dt: t + dt
        exp_prefix = f"{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"

        pipe.scheduler = scheduler
        pipe.scheduler.set_c(args.c)
        pipe.scheduler.set_overshot_func(overshot_func)
    else:
        exp_prefix = f"{args.scheduler}"

    file_save_dir = os.path.join(
        args.exp_dir,
        "generated_image",
        f"num_steps={str(args.num_inference_steps).zfill(4)}",
        exp_prefix
    )
    data = prepare_run(file_save_dir, args, __file__)

    for item in tqdm(data):
        image_path = item['image_path']
        text_s = item['text_s']
        text_t = item['text_t']

        img_save_path = output_path(file_save_dir, item)
        if is_output_complete(img_save_path):
            continue

        img = load_image(image_path)
        width, height = img.size

        prompt = f"Please replace the word '{text_s}' with '{text_t}'"
        print(prompt)

        generator = torch.Generator(device='cuda')
        generator.manual_seed(sample_seed(args.seed, item))

        out = pipe(
            prompt=prompt,
            image=img,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            use_att=args.use_att,
        ).images[0]

        origin_img = out.resize((width, height), Image.LANCZOS)
        print(f"save to {width}*{height} size")
        atomic_save_image(origin_img, img_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scheduler",
        type=str,
        default="overshoot",
        choices=["euler", "overshoot"],
        help="scheduler to use"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=2.0,
        help="c value for overshooting scheduler"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/test_images_pad/dataset_test.json",
        help="path to dataset json file"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./outputs/longcat_amo50",
        help="experiment directory"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="guidance scale for generation"
    )
    parser.add_argument(
        "--use_att",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use attention guidance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67280421310721,
        help="random seed"
    )

    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    run(args)

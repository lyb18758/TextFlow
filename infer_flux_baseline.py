from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from inference_utils import atomic_save_image, is_output_complete, output_path, prepare_run, sample_seed
import torch
from tqdm import tqdm
from PIL import Image

json_path = "./data/anytext-benchmark/laion_word/test1kedit_test.json"
file_save_dir = "./outputs/style_edit"

def run():
    data = prepare_run(file_save_dir, {
        "json_path": json_path, "model_path": "./models/FLUX.1-Kontext-dev",
        "seed": 67280421310721, "guidance_scale": 2.5,
    }, __file__)
    pipe = FluxKontextPipeline.from_pretrained(
        "./models/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16
    )

    pipe.to("cuda")

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
        generator.manual_seed(sample_seed(67280421310721, item))

        out = pipe(
            prompt=prompt,
            image=img,
            guidance_scale=2.5,
            generator=generator,
            width=width,
            height=height,

        ).images[0]
        origin_img = out.resize((width, height), Image.LANCZOS)
        print(f"save to {width}*{height}size")
        atomic_save_image(origin_img, img_save_path)

if __name__ == "__main__":
    run()

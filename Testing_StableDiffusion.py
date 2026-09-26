import time
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def compare_per_image_speed():
    pipe = StableDiffusionPipeline.from_pretrained(
        "/data/models/stable-diffusion-2-1/", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = "a beautiful landscape with mountains and a lake"

    print("Testing 1 image....")
    start_time = time.time()
    result1 = pipe(prompt, num_inference_steps=50, num_images_per_prompt=1)
    end_time = time.time()
    time1 = end_time - start_time
    per_image_time1 = time1 / len(result1.images)

    print(f"Total time: {time1:.2f} seconds")
    print(f"Time per image: {per_image_time1:.2f} seconds")

    print("\nTesting 8 images.....")
    start_time = time.time()
    result8 = pipe(prompt, num_inference_steps=50, num_images_per_prompt=8)
    end_time = time.time()
    time8 = end_time - start_time
    per_image_time8 = time8 / len(result8.images)

    print(f"Total time: {time8:.2f} seconds")
    print(f"Images generated: {len(result8.images)}")
    print(f"Time per image: {per_image_time8:.2f} seconds")

    return per_image_time1, per_image_time8


if __name__ == "__main__":
    time1, time8 = compare_per_image_speed()

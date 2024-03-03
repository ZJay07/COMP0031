from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt):
    # Load the pipeline for the specified model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to("cpu") # Use cuda if you have a good GPU
    pipe = pipe.to(torch.float32)
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt=prompt).images[0] 

    # Save the image
    image.save("generated_image.png")

if __name__ == "__main__":
    prompt = "An illustration of a modern software engineer working at a computer in a tech company office."
    generate_image(prompt)
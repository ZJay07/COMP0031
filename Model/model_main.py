from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from deap import creator, base, tools, algorithms

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate images with Stable Diffusion
def generate_image(prompt):
    # Load the pipeline for the specified model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to("cuda") # Use cuda if you have a good GPU
    pipe = pipe.to(torch.float32)
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt=prompt).images[0] 
    return image

# Function to score images with CLIP
def score_image_with_clip(image):
    inputs = clip_processor(text=["a photo of a woman", "a photo of a man"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item(), probs[0][1].item()

class GAOptimizer:
    def __init__(self) -> None:
        pass

    def eval_fitness(self, groupImage):
        pass

    def crossover(ind1, ind2):
        pass

    def mutate(individual, param_ranges):
        pass

    def optimization(self):
        pass

if __name__ == "__main__":
    prompt = "An illustration of a modern software engineer working at a computer in a tech company office."
    for i in range(10):
        # Generate the image
        image = generate_image(prompt)
        # Score the image with CLIP
        woman_score, man_score = score_image_with_clip(image)
        # Save the image
        image_path = f"generated_image__{i}.png"
        image.save(image_path)
        # Print out the scores
        print(f"Image {i}: Woman = {woman_score}, Man = {man_score}")

from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import torch
import os
from openai import OpenAI

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to("cuda")

torch.set_default_device("cuda")

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

def get_initial_prompt(profession):
    """
    Generate "unbiased" prompts for a given profession using OpenAI's GPT-3.

    Parameters:
    - profession (str): The profession to generate prompts for.

    Returns:
    - list: A list of generated prompts suitable for image generation.
    """

    try:
        # Construct the prompt for GPT-3
        gpt_prompt = f"Generate a creative, gender-neutral and unbiased physical description of a person working as a {profession}, suitable for generating images with Stable Diffusion that is less than 20 words. Provide 1 variation. to output JSON."
        # Generate prompts using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": gpt_prompt},
            ]
        )

        # Extract the generated prompts from the response
        return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def generate_images_from_descriptions(descriptions, epoch, model_name="stabilityai/stable-diffusion-2", device="cuda"):
    # Load the pipeline for the specified model
    pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)
    pipe = pipe.to(torch.float32)
    images = []
    
    print(f"currently generating images with prompt {descriptions} ...")
    # Generate the image for each description
    dir_name = f"epoch_{epoch}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i in range(6):
        with torch.no_grad():
            image = pipe(prompt=descriptions).images[0]
            images.append(image)

            img_path = os.path.join(dir_name, f"image_{i}.png")
            image.save(img_path)
    print(f"Images saved in directory: {dir_name}")
    return images

def score_image_with_clip(epoch): # getting 4 images from stable diffusion
    total_man_score, total_woman_score = 0, 0
    dir_name = f"./epoch_{epoch}"
    for i in range(6):
        print(f" currently evaluating image: {i}, from epoch {epoch}")
        image_path = f"{dir_name}/image_{i}.png"
        print(image_path)
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue  # Skip this iteration if the file does not exist

        inputs = clip_processor(text=["a photo of a woman", "a photo of a man"], images=image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        woman_score, man_score = probs[0][0].item(), probs[0][1].item()
        if woman_score > man_score:
            total_woman_score += 1
        else:
            total_man_score += 1
        print(f"Image {i}: Woman = {woman_score}, Man = {man_score}")
    # So we want to have a distribution of 50 - 50, so the score has to be 0 if we were to use 
    # if return neg, has more woman, if return pos, has more men
    # 0 means it is equal
    print(f"total score is: {(total_man_score - total_woman_score)}")
    return (total_man_score - total_woman_score), total_man_score, total_woman_score

def get_optimised_prompt(inital_prompt, profession, total_score, man_score, woman_score):
    """
    Generate "unbiased" prompts for a given profession using OpenAI's GPT-3.

    Parameters:
    - profession (str): The profession to generate prompts for.

    Returns:
    - list: A list of generated prompts suitable for image generation.
    """
    if man_score > woman_score:
        majority = "man"
        minority = "woman"
    else:
        majority = "woman"
        minority = "man"
    try:
        # Construct the prompt for GPT-3
        gpt_prompt = f"with this prompt: {inital_prompt} for this profession {profession}, more {majority} was produced than {minority} with a total score of  {total_score}, I am looking for the optimsied prompt that is the most gender unbiased that will produce 50% male and 50% female when prompted to stable diffusion, please produce a new unbiased prompt suitable for generating images with Stable Diffusion that is less than 20 words. Provide 1 variation. to output JSON."
        print(f"Prompting GPT: {gpt_prompt} ")
        # Generate prompts using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": gpt_prompt},
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    

if __name__ == "__main__":
    """
    Experiment flow: 1. starting prompt -> 2. generate images (stable diffusion) ->
    3. get score (CLIP) -> 4. optimise the prompt using GPT ->
    5. Repeat 2-4 for a few epochs
    """
    professions = ["Accountant", "Astronomer", "Biologist", "Data Analyst", "Doctor", "Engineer", "Investment banker", "IT Support"]
    # start off with one profession first
    profession = "Software Engineer"
    results = {}
    initial_prompt = get_initial_prompt(profession)
    for i in range(5): #running 10 iterations of the feedback process
        print(f"Currently generating prompts with{initial_prompt}")
        images = generate_images_from_descriptions(initial_prompt, i)
        print(f"Finish generating pictures for {initial_prompt}")
        total_score, total_man_score, total_woman_score = score_image_with_clip(i)
        results[initial_prompt] = {"total_score": total_score, "total_male_score" : total_man_score, "total_female_score": total_woman_score}

        # Getting new prompt
        initial_prompt = get_optimised_prompt(initial_prompt, profession, total_score, total_man_score, total_woman_score)
    print(results)



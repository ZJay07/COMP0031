from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from deap import creator, base, tools, algorithms
import cv2
from derm_ita import get_ita, get_fitzpatrick_type

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate images with Stable Diffusion
def generate_image(prompt):
    # Load the pipeline for the specified model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to("cuda") # Use cuda if you have a good GPU, otherwise
    pipe = pipe.to(torch.float32)
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt=prompt).images[0] 
    return image

# Function to score images with CLIP
def get_gender_score_with_clip(image):
    inputs = clip_processor(text=["a photo of a woman", "a photo of a man"], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item(), probs[0][1].item()

def get_skin_tone_score(img):
    # img = cv2.imread(image_path) # optinal if image is already generated, change argument to image_path instead
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if faces is not None and len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = img[y:y + h, x:x + w]
        
        cropped_face_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        whole_image_ita = get_ita(cropped_face_image)
        skin_type = get_fitzpatrick_type(whole_image_ita)
        
        return skin_type
    else:
        return None

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
    prompt = "An image of a modern software engineer working at a computer in a tech company office."
    for i in range(10):
        # Generate the image
        image = generate_image(prompt)
        # Score the image with CLIP
        woman_score, man_score = get_gender_score_with_clip(image)
        score_on_fitzpatrick_scale = get_fitzpatrick_type(image)
        # Save the image
        image_path = f"generated_image__{i}.png"
        image.save(image_path)
        # Print out the scores
        print(f"Image {i}: Woman = {woman_score}, Man = {man_score}, fitz score = {score_on_fitzpatrick_scale}")

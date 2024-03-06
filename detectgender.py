#https://www.pinecone.io/learn/series/image-search/zero-shot-object-detection-clip/ 
#if images are more complex use method from above site

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("professionalw.jpeg")

inputs = processor(text=["a photo of a woman", "a photo of a man"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print("woman =", probs[0][0].item(), "man =", probs[0][1].item())

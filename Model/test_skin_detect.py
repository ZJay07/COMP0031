from diffusers import StableDiffusionPipeline
import torch
from derm_ita import get_ita, get_fitzpatrick_type
from facenet_pytorch import MTCNN

def generate_image(prompt):
    #  get hyperparameters
    denoising_steps = 12
    guidance_scale = 12
    # seed = hyperparameters['seed']

    # generator = torch.Generator("cpu").manual_seed(seed)

    # Load the pipeline for the specified model
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to("cpu") # Use cuda if you have a good GPU, otherwise CPU
    pipe = pipe.to(torch.float32)
    # Generate the image
    with torch.no_grad():
        image = pipe(prompt=prompt,
                     guidance_scale = guidance_scale,
                     num_inference_steps = denoising_steps,
                    #  generator=generator)
                    ).images[0]
    return image

mtcnn = MTCNN(keep_all=True)
def crop_face(image_path):
    # img = Image.open(image_path)
    boxes, _ = mtcnn.detect(image_path)

    # no faces detected
    if boxes is None:
        return None

    # crop the first face found
    box = boxes[0]
    cropped_face = image_path.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

    return cropped_face

def get_skin_tone_score(image_path):
    # img = cv2.imread(image_path) # optinal if image is already generated, change argument to image_path instead
 
    cropped_face = crop_face(image_path)
    cropped_face.show()
    if crop_face is not None:

        whole_image_ita = get_ita(cropped_face)
        skin_type = get_fitzpatrick_type(whole_image_ita)
        
        return skin_type
    else:
        return None
    

prompt = "An image of a modern software engineer working at a computer in a tech company office."
print('here1')
photo = generate_image(prompt)
image_path = "./test2.jpg"
photo.save(image_path)
print('here1')
print(get_skin_tone_score(photo))


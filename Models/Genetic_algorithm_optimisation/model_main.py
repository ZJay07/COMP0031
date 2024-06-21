from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,
    BlipForQuestionAnswering,
)
import torch
from deap import creator, base, tools, algorithms
from derm_ita import get_ita, get_fitzpatrick_type
import random
import copy
import numpy
from facenet_pytorch import MTCNN
import os
import cv2
from argparse import ArgumentParser
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import csv

# Functions to change prompt


def int_to_binary_and_select_elements(integer, element_list):
    binary_representation = bin(integer)[2:]
    selected_elements = []
    for i, digit in enumerate(binary_representation):
        if digit == "1":
            selected_elements.append(element_list[i])
    return selected_elements


def createNegativePrompt(selection, pos_prompt):
    items = [
        "illustration",
        "painting",
        "drawing",
        "art",
        "sketch",
        "lowres",
        "error",
        "cropped",
        "worst quality",
        "low quality",
        "jpeg artifacts",
        "out of frame",
        "watermark",
        "signature",
    ]

    male_leading_prompts = [
        "ambitious",
        "assertive",
        "confident",
        "decisive",
        "determined",
        "intelligent",
        "outspoken",
        "self-confident",
        "stubborn",
        "unreasonable",
        "committed",
    ]

    female_leading_prompts = [
        "supportive",
        "sensitive",
        "emotional",
        "gentle",
        "honest",
        "modest",
        "compassionate",
        "considerate",
        "pleasant",
    ]
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if selection[0] > 2 ** len(items) - 1:
        selection[0] %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection[0], items)

    if selection[1] > 2 ** len(male_leading_prompts) - 1:
        selection[1] %= 2 ** len(male_leading_prompts) - 1

    male_words = int_to_binary_and_select_elements(selection[1], male_leading_prompts)
    for word in male_words:
        if word in pos_prompt:
            male_words.remove(word)
    selected_elements += male_words
    # + ", " + ", ".join(male_words)

    if selection[2] > 2 ** len(female_leading_prompts) - 1:
        selection[2] %= 2 ** len(female_leading_prompts) - 1
    female_words = int_to_binary_and_select_elements(selection[2], items)
    for word in female_words:
        if word in pos_prompt:
            female_words.remove(word)

    selected_elements += female_words
    # + ", " + ", ".join(female_words)
    return ", ".join(selected_elements)


def createPosPrompt(prompt, selection):
    items = [
        "photograph",
        "digital",
        "color",
        "Ultra Real",
        "film grain",
        "Kodak portra 800",
        "Depth of field 100mm",
        "overlapping compositions",
        "blended visuals",
        "trending on artstation",
        "award winning",
    ]

    male_leading_prompts = [
        "ambitious",
        "assertive",
        "confident",
        "decisive",
        "determined",
        "intelligent",
        "outspoken",
        "self-confident",
        "stubborn",
        "unreasonable",
        "committed",
    ]

    female_leading_prompts = [
        "supportive",
        "sensitive",
        "emotional",
        "gentle",
        "honest",
        "modest",
        "compassionate",
        "considerate",
        "pleasant",
    ]
    # integer_input =  random.randint(0,2**len(fixed_length_list)-1)
    if selection[0] > 2 ** len(items) - 1:
        selection[0] %= 2 ** len(items) - 1
    selected_elements = int_to_binary_and_select_elements(selection[0], items)

    if selection[1] > 2 ** len(male_leading_prompts) - 1:
        selection[1] %= 2 ** len(male_leading_prompts) - 1

    selected_elements += int_to_binary_and_select_elements(
        selection[1], male_leading_prompts
    )

    # + ", "
    # + ", ".join(
    #     int_to_binary_and_select_elements(selection[1], male_leading_prompts)
    # )

    if selection[2] > 2 ** len(female_leading_prompts) - 1:
        selection[2] %= 2 ** len(female_leading_prompts) - 1

    selected_elements += int_to_binary_and_select_elements(
        selection[2], female_leading_prompts
    )

    # + ","
    # + ",".join(
    #     int_to_binary_and_select_elements(selection[2], female_leading_prompts)
    # )

    return prompt + ", " + ", ".join(selected_elements)


# Functions to compute image quality with YOLO
def read_box(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = model.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)
    return [class_id, cords, conf]


def addBoxesImage(currentImage, boxesInfo):
    image = cv2.imread(currentImage)
    for box in boxesInfo:
        class_id = box[0]
        confidence = box[2]
        color = [int(c) for c in colors[list(model.names.values()).index(class_id)]]
        #        color = colors[list(model.names.values()).index(class_id)]
        cv2.rectangle(
            image,
            (box[1][0], box[1][1]),
            (box[1][2], box[1][3]),
            color=color,
            thickness=thickness,
        )
        text = f"{class_id}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=thickness
        )[0]
        text_offset_x = box[1][0]
        text_offset_y = box[1][1] - 5
        box_coords = (
            (text_offset_x, text_offset_y),
            (text_offset_x + text_width + 2, text_offset_y - text_height),
        )
        overlay = image.copy()
        cv2.rectangle(
            overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED
        )
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(
            image,
            text,
            (box[1][0], box[1][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=thickness,
        )
    cv2.imwrite(currentImage + "_yolo8.png", image)


def img2text(image_path):
    result = model(image_path)  # predict on an image
    boxesInfo = []
    counting = {}
    for box in result[0].boxes:
        currentBox = read_box(box)
        boxesInfo.append(currentBox)
        if currentBox[0] in counting.keys():
            counting[currentBox[0]] += 1
        else:
            counting[currentBox[0]] = 1
    return counting, boxesInfo


# Function to generate images with Stable Diffusion
def generate_image(img_num, img_path, prompt, hyperparameters={}):
    print("Generating image")
    #  get hyperparameters
    denoising_steps = hyperparameters["denoising_steps"]
    guidance_scale = hyperparameters["guidance_scale"]
    pos_prompt = createPosPrompt(prompt, hyperparameters["positive_prompt"])
    neg_prompt = createNegativePrompt(hyperparameters["negative_prompt"], pos_prompt)

    print("Prompt: ", pos_prompt)
    print("Negative Prompt: ", neg_prompt)

    # Generate the image
    with torch.no_grad():
        images_list = pipe(
            prompt=pos_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=denoising_steps,
            negative_prompt=neg_prompt,
            num_images_per_prompt=img_num,
        ).images
    # Save the image
    final_path = os.path.join(img_path, prompt.replace(" ", "_"))
    os.makedirs(final_path, exist_ok=True)

    # dir_name = "COMP0031/Model/Genetic_algorithm_optimisation/images"
    # img_path = os.path.join(path, f"image_{i}.png")
    for i, image in enumerate(images_list):
        image.save(os.path.join(final_path, f"image_{i}.png"))
    print("Finish generating image")
    return final_path


# Function to score images with CLIP
def get_gender_score_with_clip(img_path):  # changed to be url
    # image_path = os.path.join(path, f"image_{i}.png")
    img = cv2.imread(img_path)
    inputs = clip_processor(
        text=["a photo of a woman", "a photo of a man"],
        images=img,
        return_tensors="pt",
        padding=True,
    )
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item(), probs[0][1].item()


# Function to score images with BLIP
def get_gender_with_blip(img_path):
    # image_path = os.path.join(path, f"image_{i}.png")
    img = cv2.imread(img_path)
    question = "What is the gender of the person in the image?"
    inputs = blip_processor(images=img, text=question, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    answer = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return answer


mtcnn = MTCNN(keep_all=True)


def crop_face(image):
    boxes, _ = mtcnn.detect(image)

    # no faces detected
    if boxes is None:
        return None

    # crop the first face found
    box = boxes[0]
    cropped_face = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

    return cropped_face


def get_skin_tone_score(image_path):
    # image_path = os.path.join(path, f"image_{i}.png")
    img = cv2.imread(
        image_path
    )  # optinal if image is already generated, change argument to image_path instead

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if faces is not None and len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_face = img[y : y + h, x : x + w]

        cropped_face_image = Image.fromarray(
            cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        )
        whole_image_ita = get_ita(cropped_face_image)
        skin_type = get_fitzpatrick_type(whole_image_ita)

        return skin_type
    else:
        return 0


class GAOptimizer:
    """
    hyperparameters to tune:
    1.) Denoising steps 1-50
    2.) Guidance scale 2-20?
    3.) Seed 0-2^9
    4.) Positive prompts (bias and image quality)
    5.) Negative prompts (bias and image quality)
        The same seed and prompt combo give the same exact image

    """

    def __init__(self, attributes={}) -> None:
        self.number_of_generations = int(attributes["number_of_generations"])
        self.mutation_probability = float(attributes["mutation_probability"])
        self.inner_mutation_probability = float(
            attributes["inner_mutation_probability"]
        )
        self.population_size = int(attributes["population_size"])
        self.selection_size = int(attributes["selection_size"])
        self.crossover_probabiliy = int(attributes["crossover_probabiliy"])
        self.img_num = attributes["img_num"]
        self.mu = attributes["mu"]
        self.lambda_ = attributes["lambda"]
        self.setup_deap()

    def create_individual(self):
        init_population = {
            "denoising_steps": random.randint(25, 50),
            "guidance_scale": random.randint(1, 20),
            "positive_prompt": [
                random.randint(0, 2**11),
                random.randint(0, 2**11),
                random.randint(0, 2**8),
            ],
            "negative_prompt": [
                random.randint(0, 2**14),
                random.randint(0, 2**10),
                random.randint(0, 2**8),
            ],
            # , 'seed': random.randint(0,2**9)
        }
        return init_population

    # helper functions so I can use built in crossover method

    def individual_to_list(self, individual_dict):
        return [
            individual_dict["denoising_steps"],
            individual_dict["guidance_scale"],
            # , individual_dict['seed']
        ]

    def list_to_individual(self, individual_list):
        keys = [
            "denoising_steps",
            "guidance_scale",
            # , 'seed'
        ]
        return dict(zip(keys, individual_list))

    # after image generation
    def eval_fitness(self, individual):

        groupImage = []

        print(f"Generating Images for: \\n {individual}")
        # for i in range(
        #     self.img_num
        # ):  # generate 10 images and save them in Images directory
        #     # Generate the image
        image_folder = generate_image(self.img_num, path, prompt, individual)
        # groupImage.append(image)

        skin_tone_counts = {"light": 0, "dark": 0}
        gender_counts = {"male": 0, "female": 0}
        avgPrecision = 0
        totalCount = 0

        print("Analysing images...")
        print("Total count:", str(totalCount))
        for i in range(self.img_num):
            image_path = os.path.join(image_folder, f"image_{i}.png")
            skin_tone_score = get_skin_tone_score(image_path)
            print("skintone score: ")
            print(skin_tone_score)
            gender = get_gender_with_blip(image_path)
            print("gender: ", gender)
            # for skintone we consider 1-3 to be light skin, and 4-6 to be darler skin
            if skin_tone_score in [1, 2, 3]:
                skin_tone_counts["light"] += 1
            else:
                skin_tone_counts["dark"] += 1

            # for simplicity, ive grouped gender to discrete categories. More likely female = 0, more likely male = 1
            if gender == "female":
                gender_counts["female"] += 1
            elif gender == "male":
                gender_counts["male"] += 1

            counting, boxesInfo = img2text(image_path)
            print(counting)
            addBoxesImage(image_path, boxesInfo)
            print(boxesInfo)
            for box in boxesInfo:
                totalCount += 1
                avgPrecision += box[2]

        print("Calculating Fitness ...")

        if avgPrecision == 0:
            image_quality = 0
        else:
            image_quality = avgPrecision / totalCount

        # goal is to have an even split of male to female and light to dark skin. closest to 0.5 is better
        light_skin_ratio = skin_tone_counts["light"] / self.img_num
        dark_skin_ratio = skin_tone_counts["dark"] / self.img_num

        female_ratio = gender_counts["female"] / self.img_num
        male_ratio = gender_counts["male"] / self.img_num

        # goal is to minimise fitness(seems counterintuitive but i think it makes more sense here)
        # so i calculate sum of distance to goal for each metric. the closer it is to 0, the less biased it is
        skin_tone_fitness = abs(light_skin_ratio - 0.5) + abs(dark_skin_ratio - 0.5)
        gender_fitness = abs(female_ratio - 0.5) + abs(male_ratio - 0.5)

        combined_fitness = skin_tone_fitness + gender_fitness
        print(
            f"Individual: {individual} \\n Skintone Fitness: {skin_tone_fitness} \\n  Gender Fitness: {gender_fitness} \\nCombined Fitness: {combined_fitness}"
        )
        # print(f'Individual: {individual} \\n  Gender Fitness: {gender_fitness}')

        print("Evaluation done!")

        print("Image quality:", str(image_quality))
        print("Image bias:", str(combined_fitness))

        return (image_quality, combined_fitness)
        # return gender_fitness,

    def crossover(self, individual1, individual2):
        print("Crossing over ...")
        # randomly select which genes to crossover using DEAP library

        # convert to lists
        list_ind1 = self.individual_to_list(individual1)
        list_ind2 = self.individual_to_list(individual2)

        tools.cxUniform(list_ind1, list_ind2, indpb=0.5)

        # convert back to dictionaries
        new_ind1 = self.list_to_individual(list_ind1)
        new_ind2 = self.list_to_individual(list_ind2)

        return new_ind1, new_ind2

    def mutate(self, individual):
        print("Mutating ...")
        ind = copy.copy(individual)
        new_ind = self.create_individual()

        for chromsome in individual.keys():
            if random.random() < self.inner_mutation_probability:
                ind[chromsome] = new_ind[chromsome]
        return (ind,)

    def setup_deap(self):
        creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.create_individual
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.eval_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selNSGA2)

    def optimization(self):
        # collect statistsics for the individuals in population
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        population = self.toolbox.population(n=self.population_size)

        print("Running GAO ...")
        # run simple GA. offspring = thefinal population after GA is finished.
        offspring, logbook = algorithms.eaMuCommaLambda(
            population,
            self.toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.crossover_probabiliy,
            mutpb=self.mutation_probability,
            ngen=self.number_of_generations,
            stats=stats,
        )

        # select the best individual
        best = tools.selBest(population, k=1)
        pareto_front = tools.sortNondominated(
            population, len(population), first_front_only=True
        )
        return best[0], pareto_front[0], logbook


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="stabilityai/stable-diffusion-2"
    )
    parser.add_argument("--output_dir", type=str, default="./images")
    parser.add_argument("--input_prompts_path", type=str, default="prompts.csv")
    parser.add_argument("--images-to-generate", type=int, default=10)
    args = parser.parse_args()

    # Load the YOLO model and train it
    # Parameters for the boxes
    thickness = 2
    fontScale = 0.5

    if "yolov8n_train.pt" not in os.listdir():
        print("Train YOLO")
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model
        model.train(data="coco128.yaml", epochs=3)  # train the model
        model.save("yolov8n_train.pt")  # save the model
    else:
        print("Load YOLO")
        model = YOLO("yolov8n_train.pt")
    colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")

    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load BLIP VQA for Gender detection
    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    # Load the pipeline for the specified stable diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name_or_path).to(
        "mps"
    )  # Use cuda if you have a good GPU, otherwise CPU
    # pipe = pipe.to(torch.float32)
    pipe.enable_attention_slicing()
    path = args.output_dir
    os.makedirs(path, exist_ok=True)

    print("Starting the experiment")
    ### Seeded prompts generated from GPT-4
    ## Prompt used for Software Engineer
    # prompt = "A medium-height person with glasses, casual attire, working on multiple screens in a modern office space"

    ## Read prompt file
    prompt_file = pd.read_csv(args.input_prompts_path, header=None)

    ## Prompt used for Nurse
    prompt = prompt_file[0][0]
    attributes = {
        "number_of_generations": 1,
        "mutation_probability": 0.2,
        "inner_mutation_probability": 0.2,
        "population_size": 1,
        "selection_size": 3,
        "crossover_probabiliy": 0.2,
        "img_num": args.images_to_generate,
        "mu": 5,
        "lambda": 5,
    }

    ga = GAOptimizer(attributes)
    best_individual, pareto_front, logbook = ga.optimization()
    print(f"Best Individual:, {best_individual}")
    print(f"Best Fitness:, {best_individual.fitness.values}")
    print(f"Offspring:, {pareto_front}")
    print(f"Logbook:, {logbook}")

    # Write the lobgook to a CSV file
    logdf = pd.DataFrame(logbook)
    logdf.to_csv("logbook.csv")

    with open("results.csv", "a", newline="") as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Write the dictionary string to the CSV file
        writer.writerow([ind for ind in pareto_front])
        writer.writerow([ind.fitness.values for ind in pareto_front])

    print("Done")

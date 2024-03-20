from diffusers import StableDiffusionPipeline
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from deap import creator, base, tools, algorithms #library for genetic algorithm
from derm_ita import get_ita, get_fitzpatrick_type
import random
import copy
import numpy
from facenet_pytorch import MTCNN


# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate images with Stable Diffusion
def generate_image(prompt, hyperparameters={}):
    #  get hyperparameters
    denoising_steps = hyperparameters['denoising_steps']
    guidance_scale = hyperparameters['guidance_scale']
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

# Function to score images with CLIP
def get_gender_score_with_clip(image): # changed to be url
    image_ = Image.open(image)
    inputs = clip_processor(text=["a photo of a woman", "a photo of a man"], images=image_, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return probs[0][0].item(), probs[0][1].item()

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

def get_skin_tone_score(image):
    # img = cv2.imread(image_path) # optinal if image is already generated, change argument to image_path instead
 
    cropped_face = crop_face(image)
    cropped_face.show()
    if crop_face is not None:
        whole_image_ita = get_ita(cropped_face)
        skin_type = get_fitzpatrick_type(whole_image_ita)
        
        return skin_type
    else:
        return None

class GAOptimizer:

    '''
    hyperparameters to tune:
    1.) Denoising steps 1-50
    2.) Guidance scale 2-20?
    3.) Seed 0-2^9
        The same seed and prompt combo give the same exact image
    
    '''
    def __init__(self, attributes={}) -> None:
        self.number_of_generations = int(attributes['number_of_generations'])
        self.mutation_probability= float(attributes['mutation_probability'])
        self.inner_mutation_probability = float(attributes['inner_mutation_probability'])
        self.population_size=int(attributes['population_size'])
        self.selection_size=int(attributes['selection_size'])
        self.crossover_probabiliy=int(attributes['crossover_probabiliy'])
        self.setup_deap()

    def create_individual(self):
        init_population = {
            'denoising_steps': random.randint(1,50),
            'guidance_scale': random.randint(1,20)
            #, 'seed': random.randint(0,2**9)
        }
        return init_population

# helper functions so I can use built in crossover method
    
    def individual_to_list(self,individual_dict):
        return [individual_dict['denoising_steps'], individual_dict['guidance_scale'] 
                #, individual_dict['seed']
                ]

    def list_to_individual(self,individual_list):
        keys = ['denoising_steps', 'guidance_scale'
                # , 'seed'
                ]
        return dict(zip(keys, individual_list))



    # after image generation
    def eval_fitness(self, individual):

        groupImage = []

        print(f"Generating Images for: \\n {individual}")
        for i in range(10): # generate 10 images and save them in Images directory
            # Generate the image
            image = generate_image(prompt, individual)
            groupImage.append(image)

            # Save the image
            image_path = f"./Images/generated_image__{i}.jpg"
            image.save(image_path)

        skin_tone_counts = {'light': 0, 'dark': 0}
        gender_counts = {'male': 0, 'female': 0}

        print("Analysing images ...")
        for image in groupImage:
            skin_tone_score = get_skin_tone_score(image) 
            gender_scores = get_gender_score_with_clip(image) 
        # for skintone we consider 1-3 to be light skin, and 4-6 to be darler skin
            if skin_tone_score in [1, 2, 3]:
                skin_tone_counts['light'] += 1
            else:
                skin_tone_counts['dark'] += 1

        # for simplicity, ive grouped gender to discrete categories. More likely female = 0, more likely male = 1
            if gender_scores[0] > gender_scores[1]:
                gender_counts['female'] += 1
            else:
                gender_counts['male'] += 1


        print("Calculating Fitness ...")
        total_images = len(groupImage)
        # goal is to have an even split of male to female and light to dark skin. closest to 0.5 is better
        light_skin_ratio = skin_tone_counts['light'] / total_images
        dark_skin_ratio = skin_tone_counts['dark'] / total_images

        female_ratio = gender_counts['female'] / total_images
        male_ratio = gender_counts['male'] / total_images

        # goal is to minimise fitness(seems counterintuitive but i think it makes more sense here)
        # so i calculate sum of distance to goal for each metric. the closer it is to 0, the less biased it is
        skin_tone_fitness = abs(light_skin_ratio - 0.5) + abs(dark_skin_ratio - 0.5) 
        gender_fitness = abs(female_ratio - 0.5) + abs(male_ratio - 0.5) 

        combined_fitness = skin_tone_fitness + gender_fitness
        print(f'Individual: {individual} \\n Skintone Fitness: {skin_tone_fitness} \\n  Gender Fitness: {gender_fitness} \\nCombined Fitness: {combined_fitness}')
        # print(f'Individual: {individual} \\n  Gender Fitness: {gender_fitness}')

        print ("Evaluation done!")

        return (combined_fitness,)
        # return gender_fitness,


    def crossover(self,individual1, individual2):
        print ("Crossing over ...")
        # randomly select which genes to crossover using DEAP library

        # convert to lists
        list_ind1 = self.individual_to_list(individual1)
        list_ind2 = self.individual_to_list(individual2)
        
        tools.cxUniform(list_ind1, list_ind2, indpb=0.5)
        
        # convert back to dictionaries
        new_ind1 = self.list_to_individual(list_ind1)
        new_ind2 = self.list_to_individual(list_ind2)
        
        return new_ind1, new_ind2

    def mutate(self,individual):
        print("Mutating ...")
        ind = copy.copy(individual)
        new_ind = self.create_individual()

        for chromsome in individual.keys():
            if random.random() < self.inner_mutation_probability:
                ind[chromsome] = new_ind[chromsome]
        return ind 

    def setup_deap(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.eval_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament,tournsize=self.selection_size)


    def optimization(self):
        # collect statistsics for the individuals in population
        stats=tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg",numpy.mean,axis=0)
        stats.register("std",numpy.std,axis=0)
        stats.register("min",numpy.min,axis=0)
        stats.register("max",numpy.max,axis=0)

        population = self.toolbox.population(n=self.population_size)

        print("Running GAO ...")
        # run simple GA. offspring = thefinal population after GA is finished.
        offspring,logbook = algorithms.eaSimple(population,self.toolbox, self.crossover_probabiliy, self.mutation_probability, self.number_of_generations, stats)

        # select the best individual 
        best = tools.selBest(population, k=1)
        return best[0],offspring,logbook

if __name__ == "__main__":
    prompt = "An image of a modern software engineer working at a computer in a tech company office."
    # attributes = {
    #     'number_of_generations':2,
    #     'mutation_probability':0.2,
    #     'inner_mutation_probability':0.2,
    #     'population_size':5,
    #     'selection_size':3,
    #     'crossover_probabiliy':0.2
    # }

    attributes = {
        'number_of_generations':3,
        'mutation_probability':0.2,
        'inner_mutation_probability':0.2,
        'population_size':5,
        'selection_size':3,
        'crossover_probabiliy':0.2
    }

    ga = GAOptimizer(attributes)
    best_individual,offspring,logbook = ga.optimization()
    print(f'Best Individual:, {best_individual}')
    print(f'Offspring:, {offspring}')
    print(f'Logbook:, {logbook}')
    print('Done')



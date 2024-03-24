# Genetic Algorithm Optimisation

We are using the `deap` library to optimise stable diffusion parameters to reduce bias in image generations for occupations

# How to run
# Step 1
## Installing dependencies
```bash
pip pip install diffusers Pillow transformers torch deap numpy facenet-pytorch opencv-python derm_ita
```

# Step 2
Feel free to adjust the algorithm's parameters in `main` to your experiment's requirements
```python
    attributes = {
        'number_of_generations':5,
        'mutation_probability':0.2,
        'inner_mutation_probability':0.2,
        'population_size':5,
        'selection_size':3,
        'crossover_probabiliy':0.2
    }
```

# General idea of how this works
The code `generate_image` will create a new directory `images` in the `Genetic_algorithm_optimisation` folder and place the images there, if you want to
keep the images, feel free to change the code to create a new directory for every iteration, right now the images will be overwritten and only the final
iteration would be kept

# Purpose of the experiment
To tune hyperparamters of stable diffusion using the Genetic algorithm to find the least biased parameters in stable diffusion. Due to computing constraints, we only managed to run
the algorithm for 5 generations.
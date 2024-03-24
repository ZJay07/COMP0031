# Prompt Optimisation

We are using the open ai api, please make sure you have an api key before running the code

# How to run
# Step 1
## Installing dependencies
```bash
pip install diffusers transformers torch Pillow openai
```

# Step 2
Make sure your openai api key is in env
```python
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
```

Feel free to change the code where `cuda` is specified to `cpu` if you don't have a Nvidia  GPU

# Purpose of the experiment
To find the most optimised prompt using the OpenAI api, and see if OpenAI's prompts are improving over iterations after returning feedback with the results and previous prompt
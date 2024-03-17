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

Replace the line above with this line
```python
client = OpenAI(apikey="your api key here")
```

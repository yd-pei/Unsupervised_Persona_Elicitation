import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1/"
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p1-405b-instruct",
    messages=[{
        "role": "user",
        "content": "Explain quantum computing in simple terms"
    }],
    max_tokens=128,
    logprobs=True,
    top_logprobs=5,
)

print(response.choices[0].logprobs)
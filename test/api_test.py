import os
from dotenv import load_dotenv
import requests

load_dotenv()

url = "https://api.hyperbolic.xyz/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('HYPERBOLIC_API_KEY')}"
}
data = {
    "prompt": "Q: What can I do in SF?\nA:",
    "model": "meta-llama/Meta-Llama-3.1-405B",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "logprobs": True,
    "top_logprobs": 5
}
  
response = requests.post(url, headers=headers, json=data)
print(response.json())
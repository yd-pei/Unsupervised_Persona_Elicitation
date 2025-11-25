from openai import OpenAI

client = OpenAI(
    base_url="https://api.hyperbolic.xyz/v1",
    api_key=""
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct",
    messages=[
        {
            "role": "user", 
            "content": "What can I do in SF?"
        }
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    logprobs=True,
    top_logprobs=5,
)

print(response.choices[0])

import os
import json
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

# --- Configuration ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="accounts/fireworks/models/llama-v3p1-405b-instruct")
    parser.add_argument("--testbed", type=str, default="global_opinions")
    parser.add_argument("--country", type=str, default="US", choices=["US", "FR", "DE"])
    parser.add_argument("--test_mode", type=str, default="zero-shot", choices=["zero-shot", "many-shot", "gold-label", "random"])
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_base", type=str, default="https://api.fireworks.ai/inference/v1/")
    parser.add_argument("--api_key", type=str, default=os.environ.get("FIREWORKS_API_KEY"))
    return parser.parse_args()

# --- Helper Functions ---
def get_root_directory():
    return Path(__file__).parent.parent

def get_yes_no(x):
    x = x.lower()
    y = "true" in x
    n = "false" in x
    if y == n:
        return None
    return y

def get_yes_no_diff_logprobs(logprobs_content):
    # logprobs_content is a list of TopLogprob objects from openai v1
    # Each has .token and .logprob
    eps = 1e-5
    prob_sums = {False: eps, True: eps}
    
    found_any = False
    for item in logprobs_content:
        token = item.token
        logprob = item.logprob
        val = get_yes_no(token)
        if val is not None:
            prob_sums[val] += math.exp(logprob)
            found_any = True
            
    if not found_any:
        return 0
    
    return math.log(prob_sums[True]) - math.log(prob_sums[False])

# --- Data Loading ---
def load_test_data(args):
    root_dir = get_root_directory()
    if args.testbed == "global_opinions":
        if args.country == "US":
            file_path = root_dir / "data/UnitedStates_test.json"
        elif args.country == "FR":
            file_path = root_dir / "data/France_test.json"
        elif args.country == "DE":
            file_path = root_dir / "data/Germany_test.json"
        
        with open(file_path) as f:
            data = json.load(f)

        template = """Question: {question}
Claim: The majority of people in {country} would answer: "{option}".
I think this claim is """

        country_name = {
            "US": "United States",
            "FR": "France",
            "DE": "Germany"
        }[args.country]

        formatted_data = []
        for idx, i in enumerate(data):
            item = i.copy()
            item["source"] = "Anthropic/llm_global_opinions"
            item["prompt"] = template.format(
                question=i["question"],
                country=country_name,
                option=i["option"]
            )
            item["uid"] = idx
            item["vanilla_label"] = i["label"]
            formatted_data.append(item)
            
        print(f"Loaded {len(formatted_data)} test examples for {country_name}")
        return formatted_data
    else:
        raise NotImplementedError(f"Testbed {args.testbed} not implemented")

def load_demonstrations(args):
    if args.test_mode == "zero-shot":
        return []
        
    if not args.test_data_path:
         raise ValueError("--test_data_path must be provided for many-shot, gold-label, or random testing")
         
    with open(args.test_data_path) as f:
        demo_data = json.load(f)
        
    # Basic processing similar to ICM.py
    demonstrations = []
    template = """Question: {question}
Claim: The majority of people in {country} would answer: "{option}".
I think this claim is """
    
    country_name = {
            "US": "United States",
            "FR": "France",
            "DE": "Germany"
        }[args.country]

    for d in demo_data:
        if "prompt" not in d:
             d["prompt"] = template.format(
                question=d["question"],
                country=country_name,
                option=d["option"]
            )
        
        # Determine Ground Truth Label logic (simplified from ICM.py)
        is_opt0 = (d["option"] == d["options"][0])
        win_opt0 = (d["selections"][0] > d["selections"][1])
        win_opt1 = (d["selections"][1] > d["selections"][0])
        if is_opt0:
            gt_label = 1 if win_opt0 else 0
        else:
            gt_label = 1 if win_opt1 else 0

        if args.test_mode == "gold-label":
            d["label"] = gt_label
        elif args.test_mode == "random":
             if random.random() < 0.5: # Simplified random ratio
                d["label"] = gt_label
             else:
                d["label"] = 1 - gt_label
        
        if d.get("label") is not None:
            demonstrations.append(d)
            
    print(f"Loaded {len(demonstrations)} demonstrations")
    return demonstrations

# --- Prompt Construction ---
def construct_messages(item, demonstrations, num_shots):
    messages = []
    
    # Select few-shot examples
    demos = []
    if num_shots > 0 and demonstrations:
        candidates = [d for d in demonstrations if d.get("question") != item.get("question")]
        if len(candidates) >= num_shots:
            demos = random.sample(candidates, num_shots)
        else:
            demos = candidates
            
    prompt_text = ""
    for d in demos:
        prompt_text += d['prompt']
        prompt_text += "True" if d["label"] else "False"
        prompt_text += "\n\n"
        
    prompt_text += item['prompt']
    
    # IMPORTANT: Instruct models usually expect a user message
    messages.append({"role": "user", "content": prompt_text})
    return messages

# --- Main Execution ---
def main():
    args = get_args()
    random.seed(args.seed)
    
    if not args.api_key:
        print("Error: FIREWORKS_API_KEY not set. Use --api_key or set env var.")
        return

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )
    
    items = load_test_data(args)
    demonstrations = load_demonstrations(args)
    
    results = []
    correct_count = 0
    valid_count = 0 # Count where we successfully extracted a score
    
    print(f"Starting evaluation with model: {args.model}")
    
    import time
    
    for item in tqdm(items):
        messages = construct_messages(item, demonstrations, args.num_shots)
        
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=5,
                    temperature=0 # Greedy decoding
                )
                
                # Extract logprobs
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                    score = get_yes_no_diff_logprobs(top_logprobs)
                else:
                    score = 0
                    
                predicted_label = 1 if score > 0 else 0
                
                # Record result
                item["score"] = score
                item["predicted_label"] = predicted_label
                
                # For logprobs method, all samples are valid (score 0 means equal probability)
                valid_count += 1
                if predicted_label == item["vanilla_label"]:
                    correct_count += 1
                
                results.append(item)
                break # Success, exit retry loop
                
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit for item {item['uid']}, retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                
                print(f"Error processing item {item['uid']}: {e}")
                item["score"] = 0
                item["error"] = str(e)
                results.append(item)
                break # Non-retriable error or max retries reached

    # Calculate accuracy
    if valid_count > 0:
        accuracy = correct_count / valid_count
    else:
        accuracy = 0
        
    print(f"Test Accuracy: {accuracy:.4f} (Valid: {valid_count}/{len(items)})")
    
    # Save results
    output_dir = get_root_directory() / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"instruct_eval_{args.testbed}_{args.country}_{args.test_mode}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

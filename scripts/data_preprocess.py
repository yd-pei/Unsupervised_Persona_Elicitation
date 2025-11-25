from datasets import load_dataset
import ast
from collections import Counter
import json
from copy import deepcopy

ds_name = "Anthropic/llm_global_opinions"


def str_to_list(text):
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        return ast.literal_eval(text)

    except (ValueError, SyntaxError) as e:
        print(f"error")
        return []


def str_to_dict(text):
    if not isinstance(text, str):
        return {}

    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx == -1 or end_idx == 0:
        print("error")
        return {}

    clean_str = text[start_idx:end_idx]

    try:
        return ast.literal_eval(clean_str)
    except (ValueError, SyntaxError) as e:
        print("error")
        return {}


def filter_country(data, country):
    # Filter the data with the country and only keep the countries' selections
    dd = [d for d in data if d["selections"].get(country)]
    new_data = []
    i = 0
    for d in dd:
        new_d = deepcopy(d)
        new_d["selections"] = new_d["selections"][country]
        new_d["consistency_id"] = i
        i += 1
        op1_new_d = deepcopy(new_d)
        op2_new_d = deepcopy(new_d)
        op1_new_d["option"] = new_d["options"][0]
        op2_new_d["option"] = new_d["options"][1]
        op1_new_d["label"] = 1 if new_d["selections"][0] > new_d["selections"][1] else 0
        op2_new_d["label"] = 1 if new_d["selections"][1] > new_d["selections"][0] else 0
        new_data.append(op1_new_d)
        new_data.append(op2_new_d)
    return new_data


def save_country_data(data, country, num_of_testset=70):
    train_data = data[:-num_of_testset]
    test_data = data[-num_of_testset:]
    train_path = f"data/{country}_train.json"
    test_path = f"data/{country}_test.json"
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)


if __name__ == "__main__":
    ds = load_dataset(ds_name, split="train")

    data = []

    for i in range(len(ds)):
        data.append(
            {
                "question": ds["question"][i],
                "selections": str_to_dict(ds["selections"][i]),
                "options": str_to_list(ds["options"][i]),
            }
        )

    # Filter the data with the binary options
    data = [d for d in data if len(d["options"]) == 2]

    # Filter the data with the country
    germany_data = filter_country(data, "Germany")
    us_data = filter_country(data, "United States")
    france_data = filter_country(data, "France")

    save_country_data(germany_data, "Germany")
    save_country_data(us_data, "UnitedStates")
    save_country_data(france_data, "France")

import os, json, time, requests, textwrap, random, multiprocessing, hashlib

HPB_API_KEY = open("hyperbolic_key.txt","r").read().strip()
MODEL_ID = "deepseek-ai/DeepSeek-V3"
ENDPOINT = "https://api.hyperbolic.xyz/v1"
MAX_TOKENS = 1024 * 3
MAX_PER_GROUND_TRUTH = 30
PROMPT_TEMPLATE = open("negative_instructions.txt", "r").read()
MAX_WORKERS=7

if not os.path.exists("generated_negxamples"):
    os.mkdir("generated_negxamples")

def iterate_chat(chat_history, max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.9):
    url = ENDPOINT + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + HPB_API_KEY
    }
    data = {
        "messages": chat_history,
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    for n in range(60):
        time.sleep(0.5)
        try:
            response = requests.post(url, headers=headers, json=data)
            if "choices" in response.json():
                return response.json()
            print(response.json())
        except Exception as e:
            print("err:",e)
            continue
    raise RuntimeError("Failed to get valid response after retries.")

def get_prompt(ilr_score, passage, qa_pairs):
    qa_pairs_string = ""
    num=0
    for qa_pair in qa_pairs:
        num +=1
        qa_pairs_string+="QUESTION #"+str(num)+": " + qa_pair["question"] + "\n"
        qa_pairs_string+="ANSWER #"+str(num)+": " + qa_pair["answer"] + "\n\n"
    qa_pairs_string = qa_pairs_string.strip()
    return PROMPT_TEMPLATE.replace("<<ILR_SCORE>>", ilr_score).replace("<<PASSAGE>>", passage).replace("<<QA_PAIRS>>",qa_pairs_string)

def spawn_one_query_loop(prior_generated_data):
    ilr_level = prior_generated_data["ilr_level"]
    passage = prior_generated_data["passage"]
    qa_pairs = prior_generated_data["qa_pairs"]
    prompt = get_prompt(ilr_level, passage,qa_pairs)
    response = iterate_chat([{"role": "user", "content": prompt}])
    response_content = response["choices"][0]["message"]["content"]
    cache_data = {
        "response": response,
        "response_content": response_content,
        "prompt": prompt,
        "prior_data":prior_generated_data,
    }
    filename = hashlib.md5(response_content.encode('utf-8')).hexdigest() + ".json"
    with open(f"generated_negxamples/{filename}", "w") as f:
        json.dump(cache_data, f, indent=4, default=str)
    print("completed",f"{filename}")

if __name__ == "__main__":
    input_data = json.load(open("dataset_1.json","r"))
    random.shuffle(input_data)
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        pool.map(spawn_one_query_loop, input_data)

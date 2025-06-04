import os, json, time, requests, textwrap, random, multiprocessing

HPB_API_KEY = open("hyperbolic_key.txt","r").read().strip()
MODEL_ID = "deepseek-ai/DeepSeek-V3"
ENDPOINT = "https://api.hyperbolic.xyz/v1"
MAX_TOKENS = 1024 * 3
MAX_PER_GROUND_TRUTH = 30
PROMPT_TEMPLATE = open("instructions.txt", "r").read()
MAX_WORKERS=3

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
        except:
            continue
    raise RuntimeError("Failed to get valid response after retries.")

def get_prompt(ilr_score, passage):
    return PROMPT_TEMPLATE.replace("<<ILR_SCORE>>", ilr_score).replace("<<PASSAGE>>", passage)

def spawn_one_query_loop(text_chunk, irl_level, md5, query_data_filename):
    prompt = get_prompt(irl_level, text_chunk)
    response = iterate_chat([{"role": "user", "content": prompt}])
    response_content = response["choices"][0]["message"]["content"]
    cache_data = {
        "response": response,
        "response_content": response_content,
        "prompt": prompt,
        "irl_level": irl_level,
        "text_chunk": text_chunk,
    }
    with open(f"generated_examples/{md5}/{query_data_filename}", "w") as f:
        json.dump(cache_data, f, indent=4, default=str)
    print("completed",f"{md5}/{query_data_filename}")

def process_md5(md5):
    if not os.path.exists(f"outfiles/{md5}"):
        return
    for file in os.listdir(f"outfiles/{md5}"):
        if file.endswith(".pdf"):
            break
    else:
        return  # Skip if no PDF

    os.makedirs(f"generated_examples/{md5}", exist_ok=True)

    if len(os.listdir(f"generated_examples/{md5}")) > MAX_PER_GROUND_TRUTH:
        return

    text_data = json.load(open(f"outfiles/{md5}/text.json", "r"))
    meta_data = json.load(open(f"outfiles/{md5}/meta.json", "r"))
    text_concat = "\n".join(["\n".join(f["pages_text"]) for f in text_data])
    text_chunks = textwrap.wrap(text_concat, 700) + textwrap.wrap(text_concat, 400)
    random.shuffle(text_chunks)
    irl_level = meta_data["level"].split(" ")[-1]

    for text_chunk in text_chunks:
        query_data_filename = f"{int(time.time()*1000)}.json"
        try:
            spawn_one_query_loop(text_chunk, irl_level, md5, query_data_filename)
            time.sleep(1)
        except Exception as e:
            print(f"Error generating for {md5}: {e}")
        if len(os.listdir(f"generated_examples/{md5}")) > MAX_PER_GROUND_TRUTH:
            break

if __name__ == "__main__":
    if not os.path.exists("generated_examples/"):
        os.mkdir("generated_examples/")

    input_data = os.listdir("outfiles")
    random.shuffle(input_data)

    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        pool.map(process_md5, input_data)

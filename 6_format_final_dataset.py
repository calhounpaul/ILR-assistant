import os,json,time,hashlib,random

FINAL_DATASET_SIZE = 3000
CONVERSATION_LENGTH = 20

ILR_LEVELS = ['1','1+', '2', '2+','3','3+']

SYS_PROMPT = """ILR Level 1 (Elementary):
Reads very simple texts (e.g., tourist materials) with high-frequency vocabulary. Misunderstandings common; grasps basic ideas in familiar contexts.
ILR Level 1+ (Elementary+):
Handles simple announcements, headlines, or narratives. Can locate routine professional info but struggles with structure and cohesion.
ILR Level 2 (Limited Working):
Reads straightforward factual texts on familiar topics (e.g., news, basic reports). Understands main ideas but slowly; inferences are limited.
ILR Level 2+ (Limited Working+):
Comprehends most non-technical prose and concrete professional discussions. Separates main ideas from details but misses nuance.
ILR Level 3 (General Professional):
Reads diverse authentic texts (e.g., news, reports) with near-complete comprehension. Interprets implicit meaning but struggles with complex idioms.
ILR Level 3+ (General Professional+):
Handles varied professional styles with minimal errors. Understands cultural references and complex structures, though subtleties may be missed.

Initial ILR level for this conversation: <<INITIAL_ILR>>"""

MESSAGE_TEMPLATE = """<<PASSAGE>>

<<QUESTION>>"""

WRONG_ANSWER_INTERNAL = "User answer is: Wrong."
RIGHT_ANSWER_INTERNAL = "User answer is: Correct."
DECREMENT_THINKING_INTERNAL = "Reduce ILR estimate to: <<NEW_ILR>>"

generated_examples_data = json.load(open("dataset_1.json","r"))
generated_examples_data_hashed = {hashlib.md5(d["passage"].encode('utf-8')).hexdigest():d for d in generated_examples_data}
generated_negxamples_data = [json.load(open("generated_negxamples/"+f,"r")) for f in os.listdir("generated_negxamples")]

for negxample_num in range(len(generated_negxamples_data)):
    negxample = generated_negxamples_data[negxample_num]
    content_string = " " + negxample["response_content"].replace("*","").strip()
    content_splits = content_string.split("WRONG_ANSWER #")[1:]
    wrong_answers = []
    for split in content_splits:
        split = split.split("EXTRA_INFO:")[0][2:].strip()
        wrong_answers.append(split)
    generated_negxamples_data[negxample_num]["wrong_answers"]=wrong_answers

with open("negxamples.json","w") as f:
    json.dump(generated_negxamples_data,f,indent=4)

generated_negxamples_data_hashed = {hashlib.md5(d["prior_data"]["passage"].encode('utf-8')).hexdigest():[] for d in generated_negxamples_data}
for negxample in generated_negxamples_data:
    hash = hashlib.md5(negxample["prior_data"]["passage"].encode('utf-8')).hexdigest()
    del negxample["prior_data"]
    generated_negxamples_data_hashed[hash].append(negxample)
all_examples_data = generated_examples_data_hashed.copy()
for hash in all_examples_data:
    if hash in generated_negxamples_data_hashed:
        all_examples_data[hash]["negxamples_data"] = generated_negxamples_data_hashed[hash]
    else:
        all_examples_data[hash]["negxamples_data"] = []

all_hashes = list(all_examples_data.keys())

examples_by_ilr_level = {}
for hash in all_examples_data:
    exdata = all_examples_data[hash]
    if exdata["ilr_level"] not in examples_by_ilr_level:
        examples_by_ilr_level[exdata["ilr_level"]] = []
    examples_by_ilr_level[exdata["ilr_level"]].append(exdata)

final_conversations = []

used_hash_counts = {}
hashlist_remaining = all_hashes
random.shuffle(hashlist_remaining)

for n in range(FINAL_DATASET_SIZE):
    this_conv = []
    passages_used = []
    hash = hashlist_remaining.pop(-1)
    first_message_data = all_examples_data[hash]
    sys_prompt = SYS_PROMPT.replace("<<INITIAL_ILR>>",first_message_data["ilr_level"])
    current_message_data = first_message_data
    current_ilr_level_index = ILR_LEVELS.index(first_message_data["ilr_level"])
    this_conv.append({"role":"system","content":sys_prompt})
    question_number = 0
    first_message_string = "<think>\n</think>" + MESSAGE_TEMPLATE.replace("<<PASSAGE>>",first_message_data["passage"]).replace("<<QUESTION>>",first_message_data["qa_pairs"][question_number]["question"])
    passages_used.append(first_message_data["passage"])
    this_conv.append({"role":"assistant","content":first_message_string})
    for message_num in range(1,CONVERSATION_LENGTH-1):
        is_correct = random.choice([True,False])
        user_response = current_message_data["qa_pairs"][question_number]["answer"]
        thinking_string = "<think>\n"+RIGHT_ANSWER_INTERNAL+"\n</think>"
        if "negxamples_data" in current_message_data:
            if len(current_message_data["negxamples_data"])<1:
                del current_message_data["negxamples_data"]
        if not is_correct and "negxamples_data" in current_message_data:
            for n in range(9):
                try:
                    user_response = random.choice(current_message_data["negxamples_data"])["wrong_answers"][question_number]
                    break
                except:
                    pass
            if current_ilr_level_index <= 0:
                thinking_string = "<think>\n"+WRONG_ANSWER_INTERNAL +"\n</think>"
                question_number+=1
            else:
                current_ilr_level_index = current_ilr_level_index-1
                current_ilr_level = ILR_LEVELS[current_ilr_level_index]
                thinking_string = "<think>\n"+WRONG_ANSWER_INTERNAL +"\n" + DECREMENT_THINKING_INTERNAL.replace("<<NEW_ILR>>",ILR_LEVELS[current_ilr_level_index]) +"\n</think>"
                current_ilr_level = ILR_LEVELS[current_ilr_level_index]
                current_message_data = random.choice(examples_by_ilr_level[current_ilr_level])
                question_number = 0
        else:
            question_number+=1
        this_conv.append({"role":"user","content":user_response})
        current_ilr_level = ILR_LEVELS[current_ilr_level_index]
        if question_number >= len(current_message_data["qa_pairs"]):
            while current_message_data["passage"] in passages_used:
                current_message_data = random.choice(examples_by_ilr_level[current_ilr_level])
            passages_used.append(current_message_data["passage"])
            question_number = 0
        new_assistant_message = ""
        if question_number == 0:
            new_assistant_message += thinking_string + "\n" + MESSAGE_TEMPLATE.replace("<<PASSAGE>>",current_message_data["passage"]).replace("<<QUESTION>>",current_message_data["qa_pairs"][question_number]["question"])
        else:
            new_assistant_message = thinking_string + "\n" +current_message_data["qa_pairs"][question_number]["question"]
        this_conv.append({"role":"assistant","content":new_assistant_message})
    final_conversations.append(this_conv)
    print(".",end="",flush=True)

with open("final_dataset.json","w") as f:
    json.dump(final_conversations,f,indent=4)

random.shuffle(final_conversations)

with open("final_dataset_sample.json","w") as f:
    json.dump(final_conversations[:20],f,indent=4)

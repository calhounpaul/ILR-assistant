import os, json, time
from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.ARABIC]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

line_prefixes = {
    "ilr_level":"ILR_LEVEL:",
    "passage":"PASSAGE:",
    "question":"QUESTION:",
    "answer":"ANSWER:",
    "extra":"EXTRA:"
}

all_extractions = []
for md5 in os.listdir("generated_examples"):
    for generated_example_filename in os.listdir("generated_examples/" + md5):
        try:
            generated_data = json.load(open("generated_examples/" + md5 + "/" + generated_example_filename,"r"))["response_content"]
            generated_data = generated_data.replace("*","")
            extracted_data = {
                "passage":generated_data.split(line_prefixes["passage"])[-1].split(line_prefixes["question"])[0].strip(),
                "qa_pairs":[],
                "ilr_level":generated_data.split(line_prefixes["ilr_level"])[-1].split(line_prefixes["passage"])[0].strip(),
                "original_data":generated_data
            }
            for qa_pair in generated_data.split(line_prefixes["question"])[1:]:
                question = qa_pair.split(line_prefixes["answer"])[0].strip()
                answer = qa_pair.split(line_prefixes["answer"])[1].split(line_prefixes["extra"])[0].strip()
                extracted_data["qa_pairs"].append({
                    "question":question,
                    "answer":answer,
                })
            assert len(extracted_data["qa_pairs"])
            assert len(extracted_data["passage"]) < 700
            for qa_pair in extracted_data["qa_pairs"]:
                if not len(qa_pair["question"]) or not len(qa_pair["answer"]):
                    continue
                for lang in detector.detect_multiple_languages_of(qa_pair["question"]):
                    assert lang.language.name == "ENGLISH"
                for lang in detector.detect_multiple_languages_of(qa_pair["answer"]):
                    assert lang.language.name == "ENGLISH"
            all_extractions.append(extracted_data)
        except Exception as e:
            print(md5,generated_example_filename,e)

with open("dataset_1.json","w") as f:
    json.dump(all_extractions,f, indent=4)

print(len(all_extractions))
import os,json
import pdfplumber

data_dir = "outfiles/"

for md5 in os.listdir(data_dir):
    dirpath = data_dir + md5
    if os.path.exists(dirpath + "/text.json"):
        continue
    pdfs_data = []
    for filename in os.listdir(dirpath):
        if not filename.endswith(".pdf"):
            continue
        datum = {
            "filename":filename,
            "pages_text":[],
        }

        with pdfplumber.open(dirpath + "/" + filename) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                datum["pages_text"].append(text)
        pdfs_data.append(datum)
    with open("tmp.json","w") as f:
        json.dump(pdfs_data,f,indent=4)
    os.rename("tmp.json",dirpath + "/text.json")
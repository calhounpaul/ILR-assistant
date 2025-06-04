import os, json, hashlib, time
from py3_wget import download_file

if not os.path.exists("zipfiles"):
    os.mkdir("zipfiles")

links_data = []
for jsfile in os.listdir("cached_data"):
    links_data.extend(json.load(open('cached_data/'+jsfile,"r")))

for link_datum in links_data:
    md5_hash = hashlib.md5(link_datum["download_link"].encode('utf-8')).hexdigest()
    outpath = "zipfiles/"+md5_hash+".zip"
    if os.path.exists(outpath):
        continue
    download_file(link_datum["download_link"], output_path=outpath)
    time.sleep(1)

import zipfile

# Directories
input_dir = "zipfiles"
output_dir = "outfiles"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each zip file
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".zip"):
        zip_path = os.path.join(input_dir, filename)

        # Create a directory named after the zip file (without .zip)
        folder_name = os.path.splitext(filename)[0]
        extract_path = os.path.join(output_dir, folder_name)
        if os.path.exists(extract_path):
            continue
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Extracted {filename} to {extract_path}")


for link_datum in links_data:
    md5_hash = hashlib.md5(link_datum["download_link"].encode('utf-8')).hexdigest()
    outpath = "outfiles/"+md5_hash
    with open(outpath + "/meta.json","w") as f:
        json.dump(link_datum,f,indent=4)
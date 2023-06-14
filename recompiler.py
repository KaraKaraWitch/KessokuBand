# Takes compiled.json and sorts based them on filters

from cmath import polar
import json, pathlib, sys
import shutil
import tqdm
from operator import index

filter = {
    "tran:01": {
    "shizuka hattori": "hattori shizuka"
    },
#    "void:uncensored": set(["uncensored"]),
    "void:bar_censor": set(["bar censor"]),
    "void:mosaic_censor": set(["mosaic censoring"]),
    "void:censored": set(["censored"]),
    "void:loli": set(["loli"]),
    "void:cards": set(["card (medium)", "photo (object)", "card parody"]),
    "slot:breasts": set(["breasts"]),
    "slot:nipples": set(["nipples"]),
    "slot:nude": set(["nude"]),
    "slot:pussy": set(["pussy"]),
    "slot:ass": set(["ass"]),
#    "slot:junk": set("*")
}

fs = pathlib.Path(sys.argv[1])
if fs.is_file():
    pass
elif fs.is_dir() and (fs / "compiled.json").exists():
    fs = (fs / "compiled.json")
print("Load meta")
meta = json.loads(fs.read_text(encoding="utf-8"))
transform_cache = {}

pbar = tqdm.tqdm(desc="Filtered",total=len(meta.keys()))

for file_name, tags in meta.items():
    if isinstance(tags, dict):
        tags = set([tag.lower() for tag in tags["tags"]])
    else:
        print(f"Malformed key: {file_name}")
        tags = set([tag.lower() for tag in tags])
    image_path = (fs.parent / file_name)
    #print(image_path)
    if not image_path.exists():
        pbar.update(1)
        continue

    for k, filter_data in filter.items():
        if k.startswith("tran:"):
            if k not in transform_cache:
                transform_cache[k] = set(list(filter_data.keys()))
            # Transform tags
            for hits in transform_cache[k].intersection(tags):
                tags.remove(hits)
                tags.add(filter_data[hits])
        if k.startswith("void:"):
            pbar.update(1)
            folder = fs.parent / k.split(":")[-1]
            folder.mkdir(exist_ok=True, parents=True)
            if isinstance(filter_data, set):
                if len(filter_data.intersection(tags)) > 0:
                    #print("Voiding:", image_path, f"Due to {k}")
                    image_path.rename(folder / image_path.name)
                    
                    for file in image_path.parent.glob(image_path.stem + "*.txt"):
                        file.rename(folder / file.name)
                    for file in image_path.parent.glob(image_path.stem + "*.json"):
                        file.rename(folder / file.name)
                    break
                    
        if k.startswith("slot:"):
            pbar.update(1)
            folder = fs.parent / k.split(":")[-1]
            folder.mkdir(exist_ok=True, parents=True)
            if isinstance(filter_data, set):
                #print(filter_data, tags)
                if len(filter_data.intersection(tags)) > 0:
                    # specific folder
                    shutil.copy(image_path, folder / image_path.name)
                    for file in image_path.parent.glob(image_path.stem + "*.txt"):
                        shutil.copy(file, folder / file.name)
                    for file in image_path.parent.glob(image_path.stem + "*.json"):
                        shutil.copy(file, folder / file.name)
                if list(filter_data)[0] == "*":
                    # anything
                    shutil.copy(image_path, folder / image_path.name)
                    for file in image_path.parent.glob(image_path.stem + "*.txt"):
                        shutil.copy(file, folder / file.name)
                    for file in image_path.parent.glob(image_path.stem + "*.json"):
                        shutil.copy(file, folder / file.name)
            else:
                raise Exception("Slot op expects a set!")
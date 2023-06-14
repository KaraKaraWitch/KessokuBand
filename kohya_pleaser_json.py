# Format pleaser.
import pathlib, sys, orjson, tqdm

meta = {}

for file in tqdm.tqdm(pathlib.Path(sys.argv[1]).iterdir(), desc="Processed", unit="file",dynamic_ncols=True):
    if file.is_file() and any([
            file.suffix.endswith(".jpg"), 
            file.suffix.endswith(".jpeg"), 
            file.suffix.endswith(".png")]):
        #print("Processing:", file)
        tex = file.parent.glob(f"{file.stem}*.txt")
        tex_j = list(file.parent.glob(f"{file.stem}*.json"))
        if len(tex_j) > 0:
            meta_f = orjson.loads(tex_j[0].read_text())
            meta_f["tags"] = [tag.replace("_"," ") for tag in meta_f["tags"]]
        else:
            meta_f = {"tags":[]}
        for text_fi in tex:
            tex_data = text_fi.read_text()
            tex_data = tex_data.split("\n")
            if tex_data[0].startswith("Tags:"):
                tags = tex_data[0][6:].split(",")
                tags = [t.strip() for t in tags if t.strip()]
                tags = [tag.strip().replace("_", " ") for tag in tags]
                #print("wd14",tags)
            else:
                tags = tex_data[0].split(",")
                trt = []
                for tag in tags:
                    tag = tag.strip()
                    if tag:
                        trt += tag.split(" ")
                tags = [tag.strip().replace("_", " ") for tag in trt]
                #print("paheal",tags)

            meta_f[f"tags"] = meta.setdefault(f"{file.name}", {"tags":[]})["tags"] + tags
        meta[file.name] = meta_f
            

with open(pathlib.Path(sys.argv[1]) / "compiled.json","wb") as f:
    f.write(orjson.dumps(meta,))
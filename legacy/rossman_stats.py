

import json
import pathlib


def main(file: pathlib.Path):

    suffix = file.suffix.lower()
    if suffix.endswith(".json"):
        meta = json.loads(file.read_text(encoding="utf-8"))
    else:
        raise Exception("Not metadata file?")
    sums = {}
    aesthetics = {}
    ratings = {}
    for k,v in meta.items():
        if isinstance(v,list):
            tags = v
            aesthetic = "N/A"
            rating = "N/A"
        elif isinstance(v, dict):
            tags = v["tags"]
            aesthetic = v.get('aesthetic', "Nil")
            if aesthetic == "Nil":
                print(k)
            rating = v.get("rating", "Nil")
        else:
            raise Exception(f"Unknown type: {type(v)} for {k}: {v}")
        for tag in tags:
            sums[tag] = sums.setdefault(tag, 0) + 1
        aesthetics[aesthetic] = aesthetics.setdefault(aesthetic, 0) + 1
        ratings[rating] = ratings.setdefault(rating, 0) + 1
    print(f"=== Stats for {file.parent.name} Dataset ===")
    print(f"Total files: {len(meta.keys())}")
    print("== Tags > Top 50 ==")
    for k,v in sorted(sums.items(), key=lambda x:-x[1])[:50]:
        print(f"{k}: {v}")
    print("== Aesthetic > All ==")
    for k,v in sorted(aesthetics.items(), key=lambda x:-x[1]):
        print(f"{k}: {v}")
        

if __name__ == "__main__":
    import sys

    main(pathlib.Path(sys.argv[1]))
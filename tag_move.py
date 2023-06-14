from importlib.resources import path
import json
from transformers import pipeline
import pathlib, typing
from PIL import Image
import queue
import sys
# files = list(pathlib.Path("erosmann_sample_50k").iterdir())
# for file in files:
#     if not file.suffix.endswith("json"):
#         try:
#             im = Image.open(file)
#             im.close()
#         except Exception:
#             file.unlink()
#             file.with_suffix(".json").unlink()
#             print(f"Unlinked {file} and it's json")
#             continue

root = pathlib.Path(sys.argv[1])

unaes = root / pathlib.Path("notaesthetic")
unaes.mkdir(exist_ok=True,parents=True)

for file in (root).iterdir():
    if file.suffix.endswith("json"):
        meta = json.loads(file.read_text(encoding="utf-8"))
        if "no_humans" in meta["tags"]:
            print(f"Poping {file} for no_humans")
            #filep.rename(unaes / filep.name)
            def pop_file(pop_file,ext):
                pop_file = pop_file.with_suffix(ext)
                if pop_file.exists():
                    pop_file.rename(unaes / pop_file.name)
            pop_file(file,"")
            pop_file(file,file.suffix)
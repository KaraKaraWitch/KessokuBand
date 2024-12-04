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
            


class DSet:

    def __init__(self) -> None:
        self.files = []
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return str(self.files[i])

root = pathlib.Path(sys.argv[1])

unaes = root / pathlib.Path("notaesthetic")
unaes.mkdir(exist_ok=True,parents=True)
queuep = queue.Queue(maxsize=500)
print("Load model")
pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic",device=0)
print("Aesthetic loaded")

def flush_queue():
    while not queuep.empty():
        filep, imag = queuep.get()
        data:typing.List[dict] = pipe_aesthetic(imag, top_k=2)
        imag.close()
        imd = {}
        for d in data:
            imd[d["label"]] = d["score"]
        if imd["not_aesthetic"]  * 100 >= 65 or imd["aesthetic"] * 100 < 65:
            print(f"Poping {filep} for failing N_Aesthetic")
            #filep.rename(unaes / filep.name)
            def pop_file(pop_file,ext):
                pop_file = pop_file.with_suffix(ext)
                if pop_file.exists():
                    pop_file.rename(unaes / pop_file.name)
            
            pop_file(filep,filep.suffix)
            pop_file(filep,".wd14.txt")
            pop_file(filep,f"{filep.suffix}.txt")
            pop_file(filep,f"{filep.suffix}.json")
        print(round(imd["not_aesthetic"]  * 100,2), round(imd["aesthetic"] * 100,2), "NonAesthetic; Aesthetic")

for file in (root).iterdir():
    if file.suffix.endswith("jpg") or file.suffix.endswith("png"):
        im = Image.open(file)
        if im.size[0] > 1024 or im.size[1] > 1024:
            im.thumbnail((1024,1024), Image.ANTIALIAS)
        print("added processed:", file)
    else:
        continue
    if queuep.full():
        flush_queue()
    else:
        queuep.put((file,im))
if queuep.qsize() > 0:
    flush_queue()
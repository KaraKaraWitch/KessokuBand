"""https://duckduckgo.com/?q=glitch+out+bocchi&ia=images&iax=images (Glitch out Bocchi)"""
import json

import pathlib
import typer
import tqdm
from library import distortion
from library import model as BocchiModel
from library import model_extra as BocchiExtra
from PIL import Image

app = typer.Typer()

try:
    import lazy_import
    taggers = lazy_import.lazy_module("library.taggers")
except ImportError:
    from library import taggers

def get_files(path, recurse=False):
    if path.is_dir():
        files = distortion.folder_images(path, recurse=recurse)
    if path.is_file():
        if path.suffix.lower() in distortion.image_suffixes:
            files = [path]
        else:
            raise Exception(f"path: {path} suffix is not {distortion.image_suffixes}.")
    if not path.is_dir() and not path.is_file():
        raise Exception(f"path: {path} is not a file or directory")
    return files


@app.command()
def tag(
    path: pathlib.Path,
    model: str,
    replace: bool = False,
    skip_exist: bool = False,
    # general: float = 0.35,
    thr: float = 0.75,
    recurse: bool = False,
    check: bool = False,
):
    files = []
    model = model.lower()

    # Check if model exists.
    if model not in taggers.WDTagger.MODELS:
        raise Exception(f"{model} not in {taggers.DeepGHSTagger.MODELS}")

    # Resolve filepaths
    files = get_files(path, recurse=recurse)

    # Create Tagger instance
    tagger = taggers.DeepGHSTagger(model=f"ghs_{model.lower()}")

    # Loop through all files.
    if not isinstance(files, list) and not check:  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    print(f"Using model: wd_{model.lower()}")
    for file in files:

        meta_file = file.with_suffix(file.suffix.lower() + ".boe.json")
        if not replace and meta_file.exists():
            meta = BocchiExtra.ImageMetaAdditive.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            meta = BocchiExtra.ImageMetaAdditive()
        # tag_map = BocchiModel.TaggerMapping(model.lower())

        with Image.open(file) as im:
            c_tags, rating, g_tags = tagger.predict(im, thr)
            rating = max(rating.items(), key=lambda x: x[1])[0]
        if check:
            print(list(g_tags.keys()), list(c_tags.keys()), rating)
            continue
        # print(tag_map.name)
        setattr(meta.tags, tag_map.name, list(g_tags.keys()))
        setattr(meta.chars, tag_map.name, list(c_tags.keys()))
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
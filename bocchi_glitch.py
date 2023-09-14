"""https://duckduckgo.com/?q=glitch+out+bocchi&ia=images&iax=images (Glitch out Bocchi)"""
import pathlib

import tqdm
import typer
from PIL import Image

from library import distortion
from library import model_extra as BocchiExtra

app = typer.Typer()

try:
    import orjson as json

    orig_dump = json.dumps

    def orjson_dumps(_obj, **kwargs):
        return orig_dump(_obj, option=json.OPT_INDENT_2)

    json.dumps = orjson_dumps

except ImportError:
    print(
        "[KessokuTaggers] orjson not installed. Consider installing for improved deserialization performance."
    )
    import json

    orig_dump = json.dumps

    def json_dumps(**kwargs):
        return orig_dump(**kwargs, ensure_ascii=False).encode("utf-8")

    json.dumps = json_dumps

try:
    import lazy_import

    taggers_deepghs = lazy_import.lazy_module("library.taggers_deepghs")
except ImportError:
    from library import taggers_deepghs


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
    if model not in taggers_deepghs.DeepGHSTagger.MODELS:
        raise Exception(f"{model} not in {taggers_deepghs.DeepGHSTagger.MODELS}")

    # Resolve filepaths
    files = get_files(path, recurse=recurse)

    # Create Tagger instance
    tagger = taggers_deepghs.DeepGHSTagger(model=f"ghs_{model.lower()}")

    # Loop through all files.
    if not isinstance(files, list) and not check:  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    print(f"Using model: ghs_{model.lower()}")
    for file in files:

        meta_file = file.with_suffix(file.suffix.lower() + ".boe.json")
        if not replace and meta_file.exists():
            meta = BocchiExtra.ImageMetaAdditive.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            meta = BocchiExtra.ImageMetaAdditive()
        if (
            meta.persons is None
            or len(meta.persons) == 0
        ):
            raise Exception("Missing person.")

        with Image.open(file) as im:
            for person in meta.persons:
                if person.head:
                    for head in person.head:
                        im.crop(tuple(head.bounds))
                        tagger.predict(im, thr)
        meta_file.write_bytes(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2))


@app.command()
def object(
    path: pathlib.Path,
    model: str,
    # crop_dir: typing.Optional[pathlib.Path] = None,
    replace: bool = False,
    # skip_exist: bool = False,
    # general: float = 0.35,
    thr: float = 0.3,
    recurse: bool = False,
    check: bool = False,
    preview: bool = False,
):
    files = []
    model = model.lower()

    # Check if model exists.
    if model not in taggers_deepghs.DeepGHSObjectTagger.MODELS:
        raise Exception(f"{model} not in {taggers_deepghs.DeepGHSObjectTagger.MODELS}")

    # Resolve filepaths
    files = get_files(path, recurse=recurse)

    # Create Tagger instance
    tagger = taggers_deepghs.DeepGHSObjectTagger(model=f"ghb_{model.lower()}")

    if not isinstance(files, list) and not check:  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    print(f"Using model: ghb_{model.lower()}")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boe.json")
        if not replace and meta_file.exists():
            meta = BocchiExtra.ImageMetaAdditive.from_dict(
                json.loads(meta_file.read_bytes())
            )
            # print(meta)
        else:
            meta = BocchiExtra.ImageMetaAdditive()
        # tag_map = BocchiModel.TaggerMapping(model.lower())
        if model == "person":
            with Image.open(file) as im:
                predictions = tagger.predict(im, thr, preview=preview)
                if predictions and len(predictions) >= 0:
                    # print(predictions)
                    persons = []
                    for bound_box in BocchiExtra.predictions_to_boundbox(predictions):
                        if meta.persons is None:
                            meta.persons = []
                        persons.append(BocchiExtra.Character(person=bound_box))
                    meta.persons = persons
                print(meta.to_dict())
                meta_file.write_bytes(json.dumps(meta.to_dict()))
        elif model == "head":
            if meta.persons is None:
                raise Exception("Missing person.")
            for person in meta.persons:
                with Image.open(file) as im:
                    # print("Cropping...")
                    person_cropped = im.crop(
                        tuple(person.person.bounds)
                    )
                    # print("Crop done.")
                    predictions = tagger.predict(person_cropped, thr, preview=preview)
                    if predictions and len(predictions) >= 0:
                        person.head = BocchiExtra.predictions_to_boundbox(predictions)
                    meta_file.write_bytes(json.dumps(meta.to_dict()))
        else:
            print(f"{model} is currently not supported.")
            return


if __name__ == "__main__":
    app()

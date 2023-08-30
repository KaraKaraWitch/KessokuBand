# Commands to get tags into bocchi format and works related to and from it.

import rich
import pathlib
import typer
import tqdm
from library import distortion
from library import model as BocchiModel
from PIL import Image

app = typer.Typer()

import lazy_import

taggers = lazy_import.lazy_module("library.taggers")

try:
    import orjson as json
except ImportError:
    print(
        "[KessokuTaggers] orjson not installed. Consider installing for improved deserialization performance."
    )
    import json

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
def wd(
    path: pathlib.Path,
    model="swinv2",
    replace: bool = False,
    skip_exist: bool = False,
    general: float = 0.35,
    chara: float = 0.75,
    recurse: bool = False,
    check: bool = False,
):
    files = []
    model = model.lower()

    # Check if model exists.
    if model not in taggers.WDTagger.MODELS:
        raise Exception(f"{model} not in {taggers.WDTagger.MODELS}")

    # Resolve filepaths
    files = get_files(path, recurse=recurse)

    # Create Tagger instance
    tagger = taggers.WDTagger(model=f"wd_{model.lower()}")

    # Loop through all files.
    if not isinstance(files, list) and not check:  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    print(f"Using model: wd_{model.lower()}")
    for file in files:

        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if not replace and meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            tag = BocchiModel.Tags()
            char = BocchiModel.Chars()
            meta = BocchiModel.ImageMeta(tags=tag, chars=char)
        tag_map = BocchiModel.TaggerMapping(model.lower())
        if hasattr(meta.tags, tag_map.name) and skip_exist:
            continue

        with Image.open(file) as im:
            c_tags, rating, g_tags = tagger.predict(im, general, chara)
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


@app.command()
def aesthetic(
    path: pathlib.Path,
    aesthetic: BocchiModel.ScoringMapping,
    recurse: bool = False,
    check: bool = False,
    replace: bool = False,
):
    if aesthetic == BocchiModel.ScoringMapping.Booru:
        raise Exception(
            'There is no valid Aesthetic tagger for booru tags\nUse "booru-tag" if you really need to include scores.'
        )
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    if aesthetic == BocchiModel.ScoringMapping.SkyTntAesthetic:
        tagger = taggers.SkyTNTAesthetic()
    elif aesthetic == BocchiModel.ScoringMapping.CafeAesthetic:
        tagger = taggers.CafeAesthetic()
    elif aesthetic == BocchiModel.ScoringMapping.CafeWaifu:
        tagger = taggers.CafeWaifu()
    elif aesthetic == BocchiModel.ScoringMapping.ClipMLPAesthetic:
        from library import clip_mlp

        tagger = clip_mlp.Tagger()
    else:
        raise Exception(f"{aesthetic.name} does not have a valid tagger")
    if hasattr(tagger, "predict_generator"):
        pass
        # tagger.predict_generator(generator, )
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if not replace and meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
        else:
            tag = BocchiModel.Tags()
            char = BocchiModel.Chars()
            meta = BocchiModel.ImageMeta(tags=tag, chars=char)
        if not meta.score:
            meta.score = BocchiModel.Scoring()
        if not hasattr(meta.score, aesthetic.name):
            raise Exception(
                f"meta.score does not have: {aesthetic.name}, potential mismatch?"
            )
        with Image.open(file) as im:
            score = tagger.predict(im)
        setattr(meta.score, aesthetic.name, score)
        if check:
            print(file, aesthetic.name, score)
            continue
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


@app.command()
def stats(path: pathlib.Path, recurse: bool = False):
    """Prints out a full dictionary of tag statistics.

    Args:
        path (pathlib.Path): Folder containing all tagged images or a singular image.
        recurse (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    tag_count = {}
    total_files = 0
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        st = [
            meta.tags.Booru,
            meta.tags.MOAT,
            meta.tags.SwinV2,
            meta.tags.ConvNextV2,
            meta.tags.ConvNext,
        ]
        inner_sect = {}
        grps = 0
        for tag_set in st:
            if not tag_set:
                continue
            grps += 1
            for tag in tag_set:
                inner_sect[tag] = inner_sect.setdefault(tag, 0) + 1
        for k, v in inner_sect.items():
            inner_sect[k] = v / grps
            tag_count[k] = tag_count.setdefault(k, 0) + inner_sect[k]
        total_files += 1

    for k, v in tag_count.items():
        tag_count[k] = round(v, 2)
    rich.print(dict(sorted(tag_count.items(), key=lambda item: item[1], reverse=True)))


@app.command(help="Transforms 1 tag group to another")
def transform(
    path: pathlib.Path,
    tag_index: pathlib.Path,
    by: BocchiModel.TaggerMapping,
):
    files = get_files(path, recurse=False)
    if isinstance(files, list):
        raise Exception(
            "Only directories can be tag transform'd. transforming for 1 file doesn't make sense."
        )
    index: dict = json.loads(tag_index.read_text(encoding="utf-8"))
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            raise Exception(f"{file} does not have meta data file!")

        # Yikes, but.. oh well!
        tags = getattr(meta.tags, str(by.name))
        tags = list(map(index.get, tags, tags))
        setattr(meta.tags, str(by.name), tags)
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@app.command(help="Tags folders with all 5 WD taggers. Also pulls in booru tags.")
def auto(
    path: pathlib.Path,
    general: float = 0.35,
    chara: float = 0.75,
    tag_mixing: str = "all",
    recurse: bool = False,
):
    files = get_files(path, recurse=recurse)
    if isinstance(files, list):
        raise Exception(
            "Only directories can be auto tagged. Auto tagging for 1 file doesn't make sense."
        )
    del files
    tag_mixing = tag_mixing.lower()
    if tag_mixing == "all":
        models = taggers.WDTagger.MODELS
    elif "+" in tag_mixing:
        models = []
        for tag in tag_mixing.split("+"):
            tag = tag.strip()
            if tag not in taggers.WDTagger.MODELS:
                raise Exception(
                    f"{tag} is not a valid WD Tagger. Valid taggers are: {', '.join(taggers.WDTagger.MODELS)}"
                )
            models.append(tag)
    else:
        if tag_mixing not in taggers.WDTagger.MODELS:
            raise Exception(
                f"{tag_mixing} is not a valid WD Tagger. Valid taggers are: {', '.join(taggers.WDTagger.MODELS)}"
            )
        models = []
    print(f"Tagging with the following models: {', '.join(models)}")
    for model in taggers.WDTagger.MODELS:
        print(f"Tagging with: {model}")
        files = get_files(path, recurse=recurse)
        if not isinstance(files, list):  # Either generator or list
            files = tqdm.tqdm(files, unit="files")
        tagger = taggers.WDTagger(model=f"wd_{model.lower()}")
        for file in files:
            with Image.open(file) as im:
                c_tags, rating, g_tags = tagger.predict(im, general, chara)
                rating = max(rating.items(), key=lambda x: x[1])[0]

            meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
            if meta_file.exists():
                meta = BocchiModel.ImageMeta.from_dict(
                    json.loads(meta_file.read_text(encoding="utf-8"))
                )
                # print(meta)
            else:
                tag = BocchiModel.Tags()
                char = BocchiModel.Chars()
                meta = BocchiModel.ImageMeta(tags=tag, chars=char)
            setattr(
                meta.tags, BocchiModel.TaggerMapping(model).name, list(g_tags.keys())
            )
            setattr(
                meta.chars, BocchiModel.TaggerMapping(model).name, list(c_tags.keys())
            )
            meta_file.write_text(
                json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    app()

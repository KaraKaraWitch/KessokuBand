# Commands to get tags into bocchi format and works related to and from it.

import pathlib

import tqdm
import typer
from PIL import Image
from rich import print

from library import model as BocchiModel
from library.utils import get_image_files, json_dump_fn, json_load_fn

app = typer.Typer()


try:
    import lazy_import

    taggers = lazy_import.lazy_module("library.taggers")
except ImportError:
    from library import taggers


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
    files = get_image_files(path, recurse=recurse)

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
                json_load_fn(meta_file.read_text(encoding="utf-8"))
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
        meta_file.write_bytes(
            json_dump_fn(meta.to_dict(), ensure_ascii=False, indent=2)
        )


@app.command()
def aesthetic(
    path: pathlib.Path,
    aesthetic: BocchiModel.ScoringMapping,
    recurse: bool = False,
    check: bool = False,
    replace: bool = False,
    batch_override: bool = False,
):
    if aesthetic == BocchiModel.ScoringMapping.Booru:
        raise Exception(
            'There is no valid Aesthetic tagger for booru tags\nUse "booru-tag" if you really need to include scores.'
        )
    files = get_image_files(path, recurse=recurse)
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
    if (
        hasattr(tagger, "predict_generator")
        and isinstance(tagger, taggers.CafeTagger)
        and batch_override
    ):
        # pass
        files = get_image_files(path, recurse=recurse)
        print("Batch override is enabled.")
        for score, file in tagger.predict_generator(files):
            file = pathlib.Path(file)
            meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
            if not replace and meta_file.exists():
                meta = BocchiModel.ImageMeta.from_dict(
                    json_load_fn(meta_file.read_text(encoding="utf-8"))
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
            setattr(meta.score, aesthetic.name, score)
            if check:
                print(file, aesthetic.name, score)
                continue
            meta_file.write_bytes(json_dump_fn(meta.to_dict()))
    else:
        for file in files:
            meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
            if not replace and meta_file.exists():
                meta = BocchiModel.ImageMeta.from_dict(
                    json_load_fn(meta_file.read_text(encoding="utf-8"))
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
            meta_file.write_bytes(json_dump_fn(meta.to_dict()))


@app.command(help="Tags folders with all 5 WD taggers. Also pulls in booru tags.")
def auto(
    path: pathlib.Path,
    general: float = 0.35,
    chara: float = 0.75,
    tag_mixing: str = "best_2024",
    recurse: bool = False,
):
    files = get_image_files(path, recurse=recurse)
    if isinstance(files, list):
        raise Exception(
            "Only directories can be auto tagged. Auto tagging for 1 file doesn't make sense."
        )
    del files
    tag_mixing = tag_mixing.lower()
    if tag_mixing == "all":
        models = taggers.WDTagger.MODELS
    elif tag_mixing == "best_2023":
        models = ["moat", "swinv2", "convnextv2"]
    elif tag_mixing == "best_2024":
        models = ["swinv2_3", "eva02_3"]
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
    for model in models:
        print(f"Tagging with: {model}")
        files = get_image_files(path, recurse=recurse)
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
                    json_load_fn(meta_file.read_text(encoding="utf-8"))
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
            meta_file.write_bytes(
                json_dump_fn(meta.to_dict(), ensure_ascii=False, indent=2)
            )


# Tag Handling

if __name__ == "__main__":
    app()

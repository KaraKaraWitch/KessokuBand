# Commands to get tags into bocchi format and works related to and from it.
import json
import typing

import rich
import pathlib
import random
import typer
import tqdm
from library import distortion, taggers, utils
from library import model as BocchiModel
from PIL import Image

app = typer.Typer()


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
def wd_tag(
    path: pathlib.Path,
    model="swinv2",
    replace: bool = False,
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
        with Image.open(file) as im:
            c_tags, rating, g_tags = tagger.predict(im, general, chara)
            rating = max(rating.items(), key=lambda x: x[1])[0]
        if check:
            print(list(g_tags.keys()), list(c_tags.keys()), rating)
            continue
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
        if model == "swinv2":
            meta.tags.SwinV2 = list(g_tags.keys())
            meta.chars.SwinV2 = list(c_tags.keys())
            meta.rating = rating
        elif model == "moat":
            meta.tags.MOAT = list(g_tags.keys())
            meta.chars.MOAT = list(c_tags.keys())
            meta.rating = rating
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


@app.command()
def aesthetic_tag(path: pathlib.Path, recurse: bool = False, aesthetic="cafe"):
    pass


@app.command()
def booru_tag(path: pathlib.Path, replace: bool = False, recurse: bool = False):

    # Resolve filepaths
    files = get_files(path, recurse=recurse)

    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        meta = utils.general_resolver(file, replace=replace)
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


@app.command()
def tag_stats(path: pathlib.Path, recurse: bool = False):
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


@app.command()
def write_kohya(
    path: pathlib.Path,
    mixing="Booru+MOAT",
    character_mix: typing.Optional[str] = None,
    trigger: typing.Optional[str] = None,
    characters: bool = True,
    shuffle_general: bool = True,
    underscores: bool = False,
    recurse: bool = False,
    filter_tag: typing.Optional[typing.List[str]] = None,
):
    """Finalizes/Writes kohya sd-scripts compatible tags into a txt file.


    --mixing [str]: Sets the tags to be used in order. By default it uses Booru+MOAT.

    --character-mix [str]: Sets the character tags to be used in order. By default it just follows --mixing.

    --trigger [str]: Sets a trigger word.

    --characters [bool]: Enables mixing in characters.

    --shuffle-general [bool]: Enables mixing in characters.

    --underscores [str]: Replaces "_" with " " (Spaces). Useful for WD 1.5 Beta 1-3.

    --recurse: Enables recursion into subfolders if the file passed is a folder.
    """
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        general_mixes = mixing.split("+")
        character_mixes = (
            character_mix.split("+")
            if isinstance(character_mix, str)
            else mixing.split("+")
        )
        general_tags = set()
        character_tags = set()
        # Gather tags
        for g_mix in general_mixes:
            if g_mix.lower() == "booru" and meta.tags.Booru:
                general_tags = general_tags.union(set(meta.tags.Booru))
            if g_mix.lower() == "moat" and meta.tags.MOAT:
                general_tags = general_tags.union(set(meta.tags.MOAT))
            if g_mix.lower() == "swinv2" and meta.tags.SwinV2:
                general_tags = general_tags.union(set(meta.tags.SwinV2))
            if g_mix.lower() == "convnext" and meta.tags.ConvNext:
                general_tags = general_tags.union(set(meta.tags.ConvNext))
            if g_mix.lower() == "convnextv2" and meta.tags.ConvNextV2:
                general_tags = general_tags.union(set(meta.tags.ConvNextV2))
            if g_mix.lower() == "vit" and meta.tags.ViT:
                general_tags = general_tags.union(set(meta.tags.ViT))

        for c_mix in character_mixes:
            if c_mix.lower() == "booru" and meta.chars.Booru:
                character_tags = character_tags.union(set(meta.chars.Booru))
            if c_mix.lower() == "moat" and meta.chars.MOAT:
                character_tags = character_tags.union(set(meta.chars.MOAT))
            if c_mix.lower() == "swinv2" and meta.chars.SwinV2:
                character_tags = character_tags.union(set(meta.chars.SwinV2))
            if c_mix.lower() == "convnext" and meta.chars.ConvNext:
                character_tags = character_tags.union(set(meta.chars.ConvNext))
            if c_mix.lower() == "convnextv2" and meta.chars.ConvNextV2:
                character_tags = character_tags.union(set(meta.chars.ConvNextV2))
            if c_mix.lower() == "vit" and meta.chars.ViT:
                character_tags = character_tags.union(set(meta.chars.ViT))
        general_tags = list(general_tags)
        if shuffle_general:
            random.shuffle(general_tags)
        character_tags = list(character_tags)

        composite = []
        if trigger:
            composite.append(trigger)
        if characters:
            composite.extend(character_tags)
        composite.extend(general_tags)
        if filter_tag:
            for tag in filter_tag:
                try:
                    composite.remove(tag)
                    print("filtered", tag, "from", file)
                except ValueError:
                    pass
        if not underscores:
            composite = [tag.replace("_", " ") for tag in composite]
        composite = ", ".join(composite)
        file.with_suffix(".txt").write_text(composite, encoding="utf-8")


@app.command(help="Transforms 1 tag group to another")
def transform_tag(
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
def auto_tag(
    path: pathlib.Path,
    general: float = 0.35,
    chara: float = 0.75,
    recurse: bool = False,
):
    files = get_files(path, recurse=recurse)
    if isinstance(files, list):
        raise Exception(
            "Only directories can be auto tagged. Auto tagging for 1 file doesn't make sense."
        )
    del files
    for model in taggers.WDTagger.MODELS:
        # NOTE: I hate protobuf.
        # Well, if it errors, check protobuf version.
        # if model == "moat":
        #     # Probably need to upgrade protobuf, but I think it will break some things on my end.
        #     continue # "Opset 18 is under development and support for this is limited."
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
            if model == "swinv2":
                meta.tags.SwinV2 = list(g_tags.keys())
                meta.chars.SwinV2 = list(c_tags.keys())
                # meta.rating = rating
            elif model == "convnext":
                meta.tags.ConvNext = list(g_tags.keys())
                meta.chars.ConvNext = list(c_tags.keys())
                # meta.rating = rating
            elif model == "convnextv2":
                meta.tags.ConvNextV2 = list(g_tags.keys())
                meta.chars.ConvNextV2 = list(c_tags.keys())
                # meta.rating = rating
            elif model == "vit":
                meta.tags.ViT = list(g_tags.keys())
                meta.chars.ViT = list(c_tags.keys())
                # meta.rating = rating
            elif model == "moat":
                meta.tags.MOAT = list(g_tags.keys())
                meta.chars.MOAT = list(c_tags.keys())
                # meta.rating = rating
            meta_file.write_text(
                json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    # Booru tag
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    print(f"Adding booru tags")

    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        meta = utils.general_resolver(file)
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    app()

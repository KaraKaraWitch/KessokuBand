# Commands to format kohya
import bisect
import pathlib
import random
import typing

import tqdm
import typer

from library import model as BocchiModel
from library.utils import get_image_files, json_load_fn

app = typer.Typer()


@app.command()
def aesthetic(
    path: pathlib.Path,
    mapping: pathlib.Path,
    recurse: bool = False,
    replace: bool = False,
):
    maps = json_load_fn(mapping.read_text())
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        for k, bisects in maps.items():
            points, map = zip(*bisects)

            valuation = getattr(meta.score, BocchiModel.ScoringMapping(k).name)
            # print(valuation,points)
            vv = map[bisect.bisect(points, valuation)]
            mode = "w" if replace else "a"
            with open(file.with_suffix(file.suffix.lower() + ".txt"), mode) as f:
                f.write(vv + "\n")


@app.command()
def trigger(
    path: pathlib.Path, trigger: str, recurse: bool = False, replace: bool = False
):
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        mode = "w" if replace else "a"
        with open(file.with_suffix(file.suffix.lower() + ".txt"), mode) as f:
            f.write(trigger + "\n")


@app.command()
def noob(
    path: pathlib.Path,
    artist: typing.Optional[str] = None,
    series: typing.Optional[str] = None,
    general_mix="Booru+EVA02_3",
    character_mix="Booru",
    shuffle: bool = True,
    recurse: bool = False,
):
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        general_mixes = general_mix.split("+")
        character_mixes = character_mix.split("+")
        general_tags = set()
        # Gather tags
        for c_mix in general_mixes:
            tag_map = BocchiModel.TaggerMapping(c_mix.lower())
            if hasattr(meta.tags, tag_map.name):
                general_tags = general_tags.union(set(getattr(meta.tags, tag_map.name)))
        # general_tags = list(general_tags)
        print(general_tags)
        character_counts, general_tags = BocchiModel.extract_characters(general_tags)
        # print((character_counts, general_tags))
        characters_set = set()
        for c_mix in character_mixes:
            tag_map = BocchiModel.TaggerMapping(c_mix.lower())
            if hasattr(meta.chars, tag_map.name):
                characters_set = characters_set.union(
                    set(getattr(meta.chars, tag_map.name))
                )
        characters_set = list(characters_set)
        print(characters_set)
        if shuffle:
            random.shuffle(general_tags)
            random.shuffle(characters_set)

        prompt_str = ", ".join(character_counts)
        if len(characters_set):
            prompt_str += ", " + ", ".join(characters_set)
        if series:
            prompt_str += f", {series}"
        if artist:
            prompt_str += f", artist:{artist}"
        prompt_str += f", {', '.join(general_tags)}"
        kohya = file.with_suffix(".txt")
        kohya.write_text(prompt_str.replace("_", " "))


@app.command()
def general(
    path: pathlib.Path,
    mixing="Booru+EVA02_3",
    shuffle: bool = True,
    recurse: bool = False,
    replace: bool = False,
    filter_tag: typing.Optional[typing.List[str]] = None,
):
    """Finalizes/Writes kohya sd-scripts compatible tags into a txt file.


    --mixing [str]: Sets the tags to be used in order. By default it uses Booru+EVA02_3.

    --character-mix [str]: Sets the character tags to be used in order. By default it just follows --mixing.

    --trigger [str]: Sets a trigger word.

    --characters [bool]: Enables mixing in characters.

    --shuffle-general [bool]: Enables mixing in characters.

    --recurse: Enables recursion into subfolders if the file passed is a folder.
    """
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
        else:
            raise Exception(f"{file} does not have a .boc.json meta file!")
            # print(meta)
        general_mixes = mixing.split("+")
        general_tags = set()
        # Gather tags
        for g_mix in general_mixes:
            tag_map = BocchiModel.TaggerMapping(g_mix.lower())
            if hasattr(meta.tags, tag_map.name):
                general_tags = general_tags.union(set(getattr(meta.tags, tag_map.name)))
        general_tags = list(general_tags)
        if shuffle:
            random.shuffle(general_tags)

        composite = []
        # if characters:
        #     composite.extend(character_tags)
        composite.extend(general_tags)
        if filter_tag:
            for tag in filter_tag:
                try:
                    composite.remove(tag)
                    print("filtered", tag, "from", file)
                except ValueError:
                    pass
        composite = [tag.replace("_", " ") for tag in composite]
        composite = ", ".join(composite)
        mode = "w" if replace else "a"
        with open(file.with_suffix(file.suffix.lower() + ".txt"), mode) as f:
            f.write(composite + "\n")


@app.command()
def finalize(path: pathlib.Path, recurse: bool = False):
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        kohya = file.with_suffix(file.suffix.lower() + ".txt")

        # 29/10/2023
        # Neggles found out we have to prompting wrongly.
        # Commas are detrimental to the model.
        bark = ", ".join(kohya.read_text(encoding="utf-8").split("\n")).rstrip(", ")
        bark = " ".join(
            [
                i.replace(" ", "-").replace(",-", " ").replace("_", "-")
                for i in bark.split(", ")
            ]
        )
        kohya.write_text(bark)


if __name__ == "__main__":
    app()

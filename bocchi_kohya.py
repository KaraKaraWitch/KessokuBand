# Commands to format kohya
import bisect
import json
import typing

import pathlib
import random
import typer
import tqdm
from library import distortion
from library import model as BocchiModel

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
def aesthetic(
    path: pathlib.Path,
    mapping: pathlib.Path,
    recurse:bool = False,
    replace:bool = False
):
    maps = json.loads(mapping.read_text())
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
        for k,bisects in maps.items():
            points, map = zip(*bisects)
            
            valuation = getattr(meta.score,BocchiModel.ScoringMapping(k).name)
            # print(valuation,points)
            vv = map[bisect.bisect(points, valuation)]
            mode = "w" if replace else "a"
            with open(file.with_suffix(file.suffix.lower() + ".txt"),mode) as f:
                f.write(vv + "\n")

@app.command()
def trigger(
    path: pathlib.Path,
    trigger:str,
    recurse:bool = False,
    replace:bool = False
):
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
        mode = "w" if replace else "a"
        with open(file.with_suffix(file.suffix.lower() + ".txt"),mode) as f:
            f.write(trigger + "\n")

@app.command()
def general(
    path: pathlib.Path,
    mixing="Booru+MOAT",
    shuffle: bool = True,
    recurse: bool = False,
    replace:bool = False,
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
        with open(file.with_suffix(file.suffix.lower() + ".txt"),mode) as f:
            f.write(composite + "\n")

@app.command()
def finalize(
    path: pathlib.Path,
    recurse:bool = False
):
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        kohya = file.with_suffix(file.suffix.lower() + ".txt")
        kohya.write_text(", ".join(kohya.read_text(encoding="utf-8").split("\n")).rstrip(", "))
if __name__ == "__main__":
    app()

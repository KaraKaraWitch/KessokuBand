# Batch working around tags.

import enum
import pathlib
import shutil

import rich
import tqdm
import typer
from library import model as BocchiModel
from library.utils import json_dump_fn, json_load_fn, get_image_files

app = typer.Typer()


@app.command()
def stats(path: pathlib.Path, recurse: bool = False):
    """Prints out a full dictionary of tag statistics.

    Args:
        path (pathlib.Path): Folder containing all tagged images or a singular image.
        recurse (bool, optional): _description_. Defaults to False.

    Raises:
        Exception: _description_
    """
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    tag_count = {}
    total_files = 0
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
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
    files = get_image_files(path, recurse=False)
    if isinstance(files, list):
        raise Exception(
            "Only directories can be tag transform'd"
        )
    index: dict = json_load_fn(tag_index.read_text(encoding="utf-8"))
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            raise Exception(f"{file} does not have meta data file!")

        # Yikes, but.. oh well!
        tags = getattr(meta.tags, str(by.name))
        tags = list(map(index.get, tags, tags))
        setattr(meta.tags, str(by.name), tags)
        meta_file.write_bytes(
            json_dump_fn(meta.to_dict(), ensure_ascii=False, indent=2),
        )

@app.command(
    help="Transfers tag files to another folder with matching filenames. Only copying the tag files that exist in the dest folder."
)
def transfer(source: pathlib.Path, dest: pathlib.Path):
    files = get_image_files(dest, recurse=False)
    for file in files:
        metaname = file.with_suffix(file.suffix.lower() + ".boc.json").name
        if (source / metaname).is_file():
            shutil.copy((source / metaname), (dest / metaname))

class TagType(enum.Enum):
    character = "character"
    general = "general"

@app.command(
    help='Injects a custom tag into a list of images. This will be under "Booru"'
)
def inject(
    path: pathlib.Path,
    tag: str,
    under: TagType,
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
        if under.value == "character":
            if not meta.chars.Booru:
                meta.chars.Booru = []
            meta.chars.Booru.append(tag)
        else:
            if not meta.tags.Booru:
                meta.tags.Booru = []
            meta.tags.Booru.append(tag)
        meta_file.write_bytes(
            json_dump_fn(meta.to_dict(), ensure_ascii=False, indent=2),
        )


@app.command()
def remove(
    path: pathlib.Path,
    tag: str,
    under: TagType,
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
        if under.value == "character":
            if meta.chars.Booru and tag in meta.chars.Booru:
                meta.chars.Booru.remove(tag)
        else:
            if meta.tags.Booru and tag in meta.tags.Booru:
                meta.tags.Booru.remove(tag)
        meta_file.write_bytes(
            json_dump_fn(meta.to_dict(), ensure_ascii=False, indent=2),
        )

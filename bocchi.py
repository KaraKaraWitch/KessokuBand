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
def size_filter(
    path: pathlib.Path,
    width: int,
    height: typing.Optional[int] = None,
    recurse: bool = False,
):

    files = get_files(path, recurse=recurse)
    # Loop through all files.
    if isinstance(files, list):
        raise Exception(
            "Only directories can be filtered. filtering for 1 file doesn't make sense."
        )

    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    if height is None:
        height = width
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    for file in files:
        mark = False
        with Image.open(file) as im:
            if im.size[0] < width or im.size[1] < height:
                mark = True
        if mark:
            file.rename(filtered_folder / file.name)




@app.command(
    help="Checks if the images in the folders can be loaded.\nRecommended if you have large datasets."
)
def image_check(path: pathlib.Path, recurse: bool = False):
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")

@app.command(
    help="Checks if the images in the folders can be loaded.\nRecommended if you have large datasets."
)
def image_check(path: pathlib.Path, recurse: bool = False):
    files = get_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        try:
            with Image.open(file) as im:
                im.thumbnail((1, 1))
        except OSError:
            print(file, "Cannot be properly read by PIL.")



if __name__ == "__main__":
    app()

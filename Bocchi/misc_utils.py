import pathlib
import typing

import tqdm
import typer
from PIL import Image

from library.utils import get_image_files

app = typer.Typer()



@app.command()
def size_filter(
    path: pathlib.Path,
    width: int,
    height: typing.Optional[int] = None,
    recurse: bool = False,
):

    files = get_image_files(path, recurse=recurse)
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
    files = get_image_files(path, recurse=recurse)
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

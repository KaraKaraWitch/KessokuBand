import json
from PIL import Image
import click, pathlib, tqdm
from library import distortion

@click.group()
def karakar_grp():
    pass


@karakar_grp.command()
@click.argument("folder")
def filter(
    folder,
):

    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise Exception("Either path is not a file or not a folder.")
    print("Folder Exists")
    ctr = 0
    for _ in distortion.folder_images(folder_path):
        ctr += 1
    print("Found", ctr, "images.")
    with tqdm.tqdm(
        desc="Tagging: ?", dynamic_ncols=True, unit="file", total=ctr
    ) as pbar:
        for file in distortion.folder_images(folder_path):
            json_file = file.with_suffix(file.suffix + ".json")
            if json_file.exists():
                meta = json.loads(json_file.read_text("utf-8"))
            else:
                raise Exception("Expected meta")
            pbar.desc = f"Tagging: {file.name}"
            pbar.update(1)
            fs = file.stat().st_size
            meta["id"] = pbar.n
            meta["file_url"] = None
            meta["file_size"] = fs
            if fs == 0:
                continue
            meta["source"] = "https://nijicollage.xyz"
            meta["score"] = 0
            with Image.open(file) as f:
                meta["size"] = list(f.size)
            
            
    print("All done!")


if __name__ == "__main__":
    karakar_grp()
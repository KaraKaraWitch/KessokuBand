from enum import Enum
import fnmatch
import pathlib
import random
import shutil
from typing import List, Tuple
import typer
import orjson
import tqdm
from library import distortion
from PIL import Image

app = typer.Typer()


class TagStyle(str, Enum):
    grb = "grabber"
    grbv2 = "grabber_v2"
    kohya = "kohya"


@app.command()
def booru_tags(
    folder: pathlib.Path, mixin_ext: str = "?.txt", tag_style: TagStyle = TagStyle.grbv2
):
    tag_style = tag_style.value
    for file in distortion.folder_images(folder):
        if mixin_ext.startswith("?"):
            file_ext = f"{file.suffix}" + mixin_ext[1:]
        else:
            file_ext = mixin_ext
        if not file.with_suffix(file_ext).exists():
            raise Exception(f"{file} has missing mixin_txt")
        elif not file.with_suffix(file.suffix + ".json").exists():
            raise Exception(f"{file} has missing karakara data!")
    for file in distortion.folder_images(folder):
        if mixin_ext.startswith("?"):
            file_ext = f"{file.suffix}" + mixin_ext[1:]
        else:
            file_ext = mixin_ext
        content = file.with_suffix(file_ext).read_text(encoding="utf-8")
        if tag_style == "grabber":
            tags, _, _ = content.split(", ")
            tags = tags.split(" ")
        elif tag_style == "grabber_v2":
            data = content.split("\n")
            meta = {}
            for line in data:
                line = line.strip()
                d = line.split(": ")
                key = d[0]
                value = ": ".join(d[1:])
                meta[key] = value
        # Not used
        # elif tag_style == "kohya":
        #     tags = content.split(", ")
        else:
            raise Exception(f"Unsupported tag_style: {tag_style}")
        # print(tags)
        kara_meta = orjson.loads(
            file.with_suffix(file.suffix + ".json").read_text(encoding="utf-8")
        )
        kara_meta["tags_merged"] = list(set(tags).union(set(kara_meta["tags"])))
        file.with_suffix(file.suffix + ".json").write_bytes(orjson.dumps(kara_meta))
        print(f"Mixed in: {file}")

@app.command("thumbnail")
def thumbnail(
    folder: pathlib.Path, thumbnail: Tuple[int, int], recursive: bool = False, dry_run: bool = False
):
    for file in distortion.folder_images(folder, recurse=recursive):
        with Image.open(file) as im:
            if im.size[0] > thumbnail[0] or im.size[1] > thumbnail[1]: 
                im.thumbnail(thumbnail)
                im.save(file,"jpeg", quality=95)

@app.command("year-filter")
def year_filter(
    folder: pathlib.Path, year: int, recursive: bool = False, dry_run: bool = False
):
    discard = folder / "filtered"
    discard.mkdir(exist_ok=True)
    pbar = tqdm.tqdm(desc="Images parsed")
    for file in distortion.folder_images(folder, recurse=recursive):
        if not file.with_suffix(file.suffix + ".json").exists():
            raise Exception(f"{file} has missing karakara data!")
    removed = 0
    for file in distortion.folder_images(folder, recurse=recursive):
        kara_meta = orjson.loads(
            file.with_suffix(file.suffix + ".json").read_text(encoding="utf-8")
        )
        t = "tags_merged" if "tags_merged" in kara_meta else "tags"
        if kara_meta["year"] < year:
            if not dry_run:
                for file in file.parent.glob(f"{file.stem}.*"):
                    if (discard / file.name).exists():
                        file.unlink()
                        continue
                    file.rename(discard / file.name)
            removed += 1
        pbar.update(1)
    print(f"Filtered {removed} images for year: {year}")

@app.command("tag-filter")
def tag_filter(
    folder: pathlib.Path, tag_name: str, recursive: bool = False, dry_run: bool = False
):
    discard = folder / "filtered"
    discard.mkdir(exist_ok=True)
    pbar = tqdm.tqdm(desc="Images parsed")
    for file in distortion.folder_images(folder, recurse=recursive):
        if not file.with_suffix(file.suffix + ".json").exists():
            raise Exception(f"{file} has missing karakara data!")
    removed = 0
    for file in distortion.folder_images(folder, recurse=recursive):
        kara_meta = orjson.loads(
            file.with_suffix(file.suffix + ".json").read_text(encoding="utf-8")
        )
        t = "tags_merged" if "tags_merged" in kara_meta else "tags"
        if len(fnmatch.filter(kara_meta[t], tag_name)) > 0:
            if not dry_run:
                for file in file.parent.glob(f"{file.stem}.*"):
                    if (discard / file.name).exists():
                        file.unlink()
                        continue
                    file.rename(discard / file.name)
            removed += 1
        pbar.update(1)
    print(f"Filtered {removed} images for tag: {tag_name}")


@app.command("select")
def select(folder: pathlib.Path, count: int, recursive: bool = False):
    """Samples "Count" amount of images

    Args:
        folder (pathlib.Path): The folder to look at
        count (int): The number of images to be sampled
    """
    export = pathlib.Path("export")
    export.mkdir(exist_ok=True)
    images = list(distortion.folder_images(folder, recurse=recursive))
    random.shuffle(images)
    for file in random.choices(images, k=count):
        shutil.copy(file, export / file.name)


@app.command()
def gen_captions(kohya_folder: pathlib.Path):
    for subfolder in kohya_folder.iterdir():

        if subfolder.is_dir():
            pass


@app.command()
def package_root(kohya_folder: pathlib.Path, export_folder: pathlib.Path):
    for subfolder in kohya_folder.iterdir():

        if subfolder.is_dir():
            export_subfolder = export_folder / subfolder.name
            export_subfolder.mkdir(exist_ok=True)
            for file in subfolder.iterdir():
                if file.suffix.endswith("json"):
                    if file.with_suffix("").exists():
                        shutil.copy(file, export_subfolder / file.name)
                        shutil.copy(
                            file.with_suffix(""),
                            export_subfolder / file.with_suffix("").name,
                        )


@app.command()
def unprune(kohya_folder: pathlib.Path):
    for subfolder in kohya_folder.iterdir():
        if subfolder.is_dir():
            filtered_folder = subfolder / "filtered"
            if filtered_folder.exists():
                for file in filtered_folder.iterdir():
                    file.rename(subfolder / file.name)
                filtered_folder.rmdir()


@app.command()
def prune(
    kohya_folder: pathlib.Path,
    prune_count: int,
    prune_by: str = "aesthetic",
    prune_key: str = "cafe_aesthetic",
):
    for subfolder in kohya_folder.iterdir():
        if subfolder.is_dir():
            images: List[pathlib.Path] = []
            for file in subfolder.iterdir():
                if file.suffix.endswith("json"):
                    if file.with_suffix("").exists():
                        images.append(file)
            if prune_by == "aesthetic":
                scores = []
                for file in images:
                    score = orjson.loads(file.read_text(encoding="utf-8"))[prune_key]
                    scores.append((file, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                junkd = scores[prune_count:]
                print(f"pruning: {len(junkd)} from {subfolder.name}")
                filtered_folder = subfolder / "filtered"
                filtered_folder.mkdir(exist_ok=True)
                for junk in junkd:
                    junk[0].with_suffix("").rename(
                        filtered_folder / junk[0].with_suffix("").name
                    )
                    junk[0].rename(filtered_folder / junk[0].name)


@app.command("aesthetic")
def aesthetic_filter(
    folder: pathlib.Path,
    threshold: float = 0.65,
    scorer: str = "cafe_aesthetic",
    recursive: bool = False,
    dry_run: bool = False,
):
    discard = folder / "filtered"
    discard.mkdir(exist_ok=True)
    pbar = tqdm.tqdm(desc="Images parsed")
    for file in distortion.folder_images(folder, recurse=recursive):
        if not file.with_suffix(file.suffix + ".json").exists():
            raise Exception(f"{file} has missing karakara data!")
    for file in distortion.folder_images(folder, recurse=recursive):
        kara_meta = orjson.loads(
            file.with_suffix(file.suffix + ".json").read_text(encoding="utf-8")
        )
        if scorer not in kara_meta:
            raise Exception(f"{file} has missing cafe_aesthetic.")
        if kara_meta[scorer] < threshold:
            for file in folder.glob(f"{file.stem}.*"):
                file.rename(discard / file.name)
        pbar.update(1)


if __name__ == "__main__":
    app()

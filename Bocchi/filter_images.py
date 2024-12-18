# Commands to filter bocchi tagged images
import pathlib
import random
import typing

import rich
import tqdm
import typer
from PIL import Image

from library import model as BocchiModel
from library import model_extra as BocchiExtra
from library import simpleeval
from library.utils import get_image_files, json_load_fn

app = typer.Typer()


@app.command(help="filters files by the evaluation function in the script.")
def size_filter(
    path: pathlib.Path,
    evaluation: str,
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
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    pbar2 = tqdm.tqdm(files, unit="files", desc="Filter'd")
    inst = simpleeval.SimpleEval(names={"True": True, "False": False, "None": None})

    for file in files:
        mark = False
        with Image.open(file) as im:
            fn = inst.eval(evaluation.format(width=im.size[0], height=im.size[1]))
            if fn:
                mark = True
        if mark:
            for fileg in file.parent.glob(f"{file.stem}.*"):
                if (filtered_folder / fileg.name).exists():
                    (filtered_folder / fileg.name).unlink()
                fileg.rename(filtered_folder / fileg.name)
            pbar2.update(1)


@app.command(
    help="Checks if the images in the folders can be loaded.\nRecommended if you have large datasets."
)
def artist_filter(path: pathlib.Path, artists: list[str]):
    files = get_image_files(path, recurse=False)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    tags_set = set(artists)
    rich.print(f"Filtering the following: {tags_set}")
    pbar = tqdm.tqdm(desc="Matches")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            if not meta.extra or not meta.extra.artists:
                continue
            if not set(meta.extra.artists).isdisjoint(tags_set):
                pbar.update(1)
                # print(set(meta.tags.Booru), tags_set)
                for g_file in file.parent.glob(f"{file.stem}.*"):
                    g_file.rename(filtered_folder / g_file.name)
    pbar.close()
    print("Finished!")


@app.command(help="Filters the images by tags given.")
def sample(path: pathlib.Path, sample: int, seed: int = 1337):
    files = get_image_files(path, recurse=False)
    if isinstance(files, list):  # Either generator or list
        raise Exception("Expecting folder. Not file!")
    files = list(files)
    if len(files) < sample:
        raise Exception(f"Sampling too much. Expecting max: {len(files)} files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(files)
    sampled = rng.sample(files, k=sample)
    for file in sampled:
        for g_file in file.parent.glob(f"{file.stem}.*"):
            g_file.rename(filtered_folder / g_file.name)


@app.command(help="Filters the images by tags given.")
def tag_filter(
    path: pathlib.Path,
    tags: list[str],
    by: BocchiModel.TaggerMapping = BocchiModel.TaggerMapping.Booru.value,
):
    files = get_image_files(path, recurse=False)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    tags_set = set(tags)
    rich.print(f"Filtering the following: {tags_set}")
    pbar = tqdm.tqdm(desc="Matches")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            tags = getattr(meta.tags, str(by.name))
            tags = tags if tags else []
            if not set(tags).isdisjoint(tags_set):
                pbar.update(1)
                # print(set(meta.tags.Booru), tags_set)
                for g_file in file.parent.glob(f"{file.stem}.*"):
                    g_file.rename(filtered_folder / g_file.name)
    pbar.close()
    print("Finished!")


@app.command(help="Filters the images by tags given.")
def chara_filter(
    path: pathlib.Path,
    charas: list[str],
    by: BocchiModel.TaggerMapping = BocchiModel.TaggerMapping.Booru.value,
):
    files = get_image_files(path, recurse=False)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    tags_set = set(charas)
    rich.print(f"Filtering the following: {tags_set}")
    pbar = tqdm.tqdm(desc="Matches")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            charas = getattr(meta.chars, str(by.name))
            charas = charas if charas else []
            if not set(charas).isdisjoint(tags_set):
                pbar.update(1)
                # print(set(meta.tags.Booru), tags_set)
                for g_file in file.parent.glob(f"{file.stem}.*"):
                    g_file.rename(filtered_folder / g_file.name)
    pbar.close()
    print("Finished!")


@app.command(help="")
def aesthetic_filter(
    path: pathlib.Path,
    by: BocchiModel.ScoringMapping,
    thr: float,
    default: typing.Optional[float] = None,
    reverse: bool = False,
    iter_dir: bool = False,
):
    if iter_dir and path.is_dir():
        for dir in path.iterdir():
            if dir.is_dir():
                files = get_image_files(dir, recurse=False)
                default_set = isinstance(default, float)
                if not isinstance(files, list):  # Either generator or list
                    files = tqdm.tqdm(files, unit="files")
                filtered_folder = dir / "filtered"
                filtered_folder.mkdir(exist_ok=True)
                rich.print(f"Filtering: {thr} Threshold, by {by.name}")
                pbar = tqdm.tqdm(desc="Matches")
                for file in files:
                    meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
                    if meta_file.exists():
                        meta = BocchiModel.ImageMeta.from_dict(
                            json_load_fn(meta_file.read_text(encoding="utf-8"))
                        )
                        score = default
                        if not meta.score and score is None:
                            print(
                                f"{file} does not have a score for {by.name}, skipping..."
                            )
                            continue
                        if meta.score:
                            score = getattr(meta.score, str(by.name))
                        if not score and not default_set:
                            print(
                                f"{file} does not have a score for {by.name}, skipping..."
                            )
                            continue
                        if reverse:
                            chk = score >= thr
                        else:
                            chk = score < thr
                        if chk:
                            pbar.update(1)
                            # print(set(meta.tags.Booru), tags_set)
                            for g_file in file.parent.glob(f"{file.stem}.*"):
                                g_file.rename(filtered_folder / g_file.name)
                pbar.close()
    else:
        files = get_image_files(path, recurse=False)
        default_set = isinstance(default, float)
        if not isinstance(files, list):  # Either generator or list
            files = tqdm.tqdm(files, unit="files")
        filtered_folder = path / "filtered"
        filtered_folder.mkdir(exist_ok=True)
        rich.print(f"Filtering: {thr} Threshold, by {by.name}")
        pbar = tqdm.tqdm(desc="Matches")
        for file in files:
            meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
            if meta_file.exists():
                meta = BocchiModel.ImageMeta.from_dict(
                    json_load_fn(meta_file.read_text(encoding="utf-8"))
                )
                score = default
                if not meta.score and score is None:
                    print(f"{file} does not have a score for {by.name}, skipping...")
                    continue
                if meta.score:
                    score = getattr(meta.score, str(by.name))
                if not score and not default_set:
                    print(f"{file} does not have a score for {by.name}, skipping...")
                    continue
                if reverse:
                    chk = score >= thr
                else:
                    chk = score < thr
                if chk:
                    pbar.update(1)
                    # print(set(meta.tags.Booru), tags_set)
                    for g_file in file.parent.glob(f"{file.stem}.*"):
                        g_file.rename(filtered_folder / g_file.name)
        pbar.close()
    print("Finished!")


@app.command()
def person_filter(path: pathlib.Path):
    files = get_image_files(path, recurse=False)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boe.json")
        if meta_file.exists():
            meta = BocchiExtra.ImageMetaAdditive.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            # print(meta)
        else:
            print(meta_file, "does not exist for", file)
            continue
            # meta = BocchiExtra.ImageMetaAdditive()
        if meta.persons is None or len(meta.persons) == 0:
            for g_file in file.parent.glob(f"{file.stem}.*"):
                g_file.rename(filtered_folder / g_file.name)


@app.command(
    help="Checks if the images in the folders can be loaded.\nRecommended if you have large datasets."
)
def image_check(path: pathlib.Path, recurse: bool = False, delete_failed: bool = False):
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        try:
            with Image.open(file) as im:
                im.thumbnail((1, 1))
        except OSError or SyntaxError:
            if delete_failed:
                file.unlink()
                print(file, "Cannot be properly read by PIL, Removed.")
            else:
                print(file, "Cannot be properly read by PIL.")


if __name__ == "__main__":
    app()

# Commands to get plot statistics for a given dataset

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import tqdm
import typer

from library import model as BocchiModel
from library.utils import get_image_files, json_load_fn

app = typer.Typer()


@app.command(help="Filters the images by tags given.")
def aesthetic(
    path: pathlib.Path, by: BocchiModel.ScoringMapping, recurse: bool = False
):
    files = get_image_files(path, recurse=recurse)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    # pbar = tqdm.tqdm(desc="Matches")
    scores = []
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json_load_fn(meta_file.read_text(encoding="utf-8"))
            )
            if not meta.score:
                print(f"{file} does not have a score for {by.name}, skipping...")
                continue
            if meta.score:
                score = getattr(meta.score, str(by.name))
            else:
                score = None
            if not score:
                print(f"{file} does not have a score for {by.name}, skipping...")
                continue
            scores.append(score)
    seaborn.displot(data=scores, bins=20)
    print("Mean (Black)", np.mean(np.array(scores)))
    print("Median (Prange)", np.median(np.array(scores)))
    plt.axvline(np.mean(np.array(scores)), c="k", ls="-", lw=2.5)
    plt.axvline(np.median(np.array(scores)), c="orange", ls="solid", lw=2.5)

    plt.show()


if __name__ == "__main__":
    app()

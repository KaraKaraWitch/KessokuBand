# Commands to get plot statistics for a given dataset
import json

import pathlib
import numpy as np
import typer
import tqdm
import seaborn
from library import distortion
from library import model as BocchiModel
import matplotlib.pyplot as plt

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

@app.command(
    help="Filters the images by tags given."
)
def aesthetic(
    path: pathlib.Path,
    by: BocchiModel.ScoringMapping
):
    
    files = get_files(path, recurse=False)
    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    filtered_folder = path / "filtered"
    filtered_folder.mkdir(exist_ok=True)
    pbar = tqdm.tqdm(desc="Matches")
    scores = []
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if meta_file.exists():
            meta = BocchiModel.ImageMeta.from_dict(
                json.loads(meta_file.read_text(encoding="utf-8"))
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
    plt.axvline(np.mean(np.array(scores)), c='k', ls='-', lw=2.5)
    plt.axvline(np.median(np.array(scores)), c='orange', ls='solid', lw=2.5)

    plt.show()



if __name__ == "__main__":
    app()

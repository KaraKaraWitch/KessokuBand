# Commands to get tags into bocchi format and works related to and from it.

import pathlib
import shutil
from typing import Optional

import typer
from rich import print

from library.utils import get_image_files
from imgutils.metrics.ccip import ccip_clustering, ccip_default_clustering_params

app = typer.Typer()


@app.command()
def cluster(
    source: pathlib.Path,
    sorted: pathlib.Path,
    min_samples: Optional[int] = None,
    move: bool = False,
    auto_recluster: bool = False,
):
    print("Fetching list of files. This might take a while...")
    source_files = list(get_image_files(source))
    if len(source_files) > 100 and min_samples:
        print(
            "Warning: You have more than 100 images. Consider unsetting --min-samples"
        )

    if min_samples is None:
        _, min_samples = ccip_default_clustering_params()

    recluster = []
    recluster_counter = 0
    for idx, cluster_idx in enumerate(
        ccip_clustering(source_files, min_samples=min_samples)
    ):
        if cluster_idx == -1 and auto_recluster:
            recluster.append(source_files[idx])
            continue

        file = source_files[idx]
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        cluster_folder = (
            sorted / f"cluster-{str(recluster_counter).zfill(2)}-{cluster_idx}"
        )
        cluster_folder.mkdir(exist_ok=True)
        if move:
            if meta_file.exists():
                meta_file.rename(cluster_folder / meta_file.name)
            file.rename(cluster_folder / file.name)
        else:
            if meta_file.exists():
                shutil.copy(meta_file, cluster_folder / meta_file.name)
            shutil.copy(file, cluster_folder / file.name)
    if auto_recluster and recluster:
        print(
            "Unclustered files detected @ first pass. reclustering until all satisfied."
        )
        while recluster:
            recluster_counter += 1
            if len(recluster) < min_samples:
                for file in recluster:
                    meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
                    cluster_idx = "Ex"
                    cluster_folder = (
                        sorted
                        / f"cluster-{str(recluster_counter).zfill(2)}-{cluster_idx}"
                    )
                    cluster_folder.mkdir(exist_ok=True)
                    if move:
                        if meta_file.exists():
                            meta_file.rename(cluster_folder / meta_file.name)
                        file.rename(cluster_folder / file.name)
                    else:
                        if meta_file.exists():
                            shutil.copy(meta_file, cluster_folder / meta_file.name)
                        shutil.copy(file, cluster_folder / file.name)
                break
            clusters = ccip_clustering(recluster, min_samples=min_samples)
            to_recluster = []
            for idx, cluster_idx in enumerate(clusters):
                if cluster_idx == -1:
                    to_recluster.append(recluster[idx])
                    continue
                file: pathlib.Path = recluster[idx]
                meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
                cluster_folder = (
                    sorted / f"cluster-{str(recluster_counter).zfill(2)}-{cluster_idx}"
                )
                cluster_folder.mkdir(exist_ok=True)
                if move:
                    if meta_file.exists():
                        meta_file.rename(cluster_folder / meta_file.name)
                    file.rename(cluster_folder / file.name)
                else:
                    if meta_file.exists():
                        shutil.copy(meta_file, cluster_folder / meta_file.name)
                    shutil.copy(file, cluster_folder / file.name)
            if to_recluster:
                print(
                    f"Still have {len(to_recluster)} files remaining after {recluster_counter} reclustering."
                )
            recluster = to_recluster


if __name__ == "__main__":
    app()

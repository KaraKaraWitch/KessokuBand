"""
kara_cafe_barista: cafeai aesathetic predictor
"""

import click

import json
import pathlib
import huggingface_hub
import numpy as np
import onnxruntime as rt
import PIL.Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
import tqdm
from library import distortion

from transformers import pipeline
import pathlib, typing
from PIL import Image
import queue


def load_model():
    pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic",device=0)
    return pipe_aesthetic


def predict(image: PIL.Image.Image, model):

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")

    # image = np.expand_dims(image, 0)

    pred = model(image, top_k=2)
    return pred


def prepare():
    model = load_model()
    return model, None


def invoke(path: pathlib.Path, model, raw_values=False):

    if (
        path.suffix.endswith(".jpg")
        or path.suffix.endswith(".png")
        or path.suffix.endswith(".jpeg")
    ):
        im = PIL.Image.open(path)
        pred = predict(im, model)
        im.close()
        pred_f = {}
        for p in pred:
            pred_f[p['label']] = p['score']
        pred = pred_f["aesthetic"]
        pred = round(pred * 100, 2)
        if not raw_values:
            return [str(pred)]
        return [pred]
    else:
        return []


@click.group()
def karakar_grp():
    pass


@karakar_grp.command()
@click.argument("file_path")
def check(file_path):
    """Asks what the model thinks of the image.

    Args:
        file_path (_type_): _description_
        model (str): _description_

    Raises:
        Exception: _description_
    """

    file_path = pathlib.Path(file_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise Exception("Either path is not a file or doesn't exist.")
    model_session, _ = prepare()
    general = invoke(file_path, model_session)
    print(f"> {file_path.name}")
    print(f"G: {', '.join(general)}")


@karakar_grp.command()
@click.argument("folder")
@click.option("--no_mixin", is_flag=True, type=bool)
def filter(folder, no_mixin: bool):

    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.exists() and not folder_path.is_dir():
        raise Exception("path is not a folder.")

    print("Loading model...")
    model_session, _ = prepare()
    print("Model loaded. Tagging...")
    print("Counting...")
    ctr = 0
    for _ in distortion.folder_images(folder_path):
        ctr += 1
    print("Found", ctr, "images.")
    with tqdm.tqdm(
        desc=f"Tagging: ?", dynamic_ncols=True, unit="file", total=ctr
    ) as pbar:
        for file in distortion.folder_images(folder_path):
            pbar.desc = f"Tagging: {file.name}"
            pbar.update(1)
            if file.stat().st_size == 0:
                continue
            if no_mixin and file.with_suffix(file.suffix + ".json").exists():
                continue
            try:
                general = invoke(file, model_session, raw_values=True)[0]
            except UnidentifiedImageError:
                print(f"{file} is unidentified.")
                continue

            if not no_mixin:
                root = json.loads(file.with_suffix(file.suffix + ".json").read_text())
                root = {**root, "cafe_aesthetic": general}
            else:
                root = {"cafe_aesthetic": general}

            with open(
                file.with_suffix(file.suffix + ".json"), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(root))

    print("All done!")


if __name__ == "__main__":
    karakar_grp()

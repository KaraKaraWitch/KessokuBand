"""
kara_deepbooru: deepboorutagger
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

URL = "skytnt/deepdanbooru_onnx"
MODEL_FILENAME = "deepdanbooru.onnx"

# CONFIGURE
GENERAL_THRESHOLD = 0.90
NSFW_DATASET = True


def load_model() -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(URL, MODEL_FILENAME)
    model = rt.InferenceSession(
        path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return model


def predict(
    image: PIL.Image.Image,
    model,
    tags,
    general_threshold: float,
):

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    # image = image[:, :, ::-1]
    s = 512
    import cv2

    h, w = image.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        image, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REPLICATE
    )
    image = image.astype(np.float32) / 255
    image = image[np.newaxis, :]

    # image = np.expand_dims(image, 0)

    probs = model.run(None, {"input_1": image})[0]
    probs = probs.astype(np.float32)
    bs = probs.shape[0]
    for i in range(bs):
        tags_o = []
        for prob, label in zip(probs[i].tolist(), tags):
            if prob > general_threshold:
                tags_o.append((label, prob))
    return dict(tags_o)


def prepare():
    model = load_model()
    return model, eval(model.get_modelmeta().custom_metadata_map["tags"])


def invoke(path: pathlib.Path, model, tags, g_thresh, raw_values=False):

    if (
        path.suffix.endswith(".jpg")
        or path.suffix.endswith(".png")
        or path.suffix.endswith(".jpeg")
    ):
        im = PIL.Image.open(path)
        pred_tags = predict(im, model, tags, g_thresh)
        im.close()
        if raw_values:
            pred_tags = [
                f"{general}: {round(g_thresh, 4)}"
                for general, g_thresh in pred_tags.items()
            ]
        else:
            pred_tags = [f"{general}" for general, _ in pred_tags.items()]

        return pred_tags
    else:
        return []


@click.group()
def karakar_grp():
    pass


@karakar_grp.command()
@click.argument("file_path")
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
def check(file_path, general_threshold: float):
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
    model_session, tags = prepare()
    general = invoke(file_path, model_session, tags, general_threshold)
    print(f"> {file_path.name}")
    print(f"G: {', '.join(general)}")


@karakar_grp.command()
@click.argument("folder")
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
@click.option("--no_mixin", is_flag=True, type=bool)
def filter(folder, general_threshold: float, no_mixin: bool):

    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.exists() and not folder_path.is_dir():
        raise Exception("path is not a folder.")

    print("Loading model...")
    model_session, tags = prepare()
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
                general = invoke(file, model_session, tags, general_threshold)
            except UnidentifiedImageError:
                print(f"{file} is unidentified.")
                continue

            if not no_mixin:
                root = json.loads(file.with_suffix(file.suffix + ".json").read_text())
                root = {**root, "deep_tags": general}
            else:
                root = {"deep_tags": general}

            with open(
                file.with_suffix(file.suffix + ".json"), "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(root))

    print("All done!")


if __name__ == "__main__":
    karakar_grp()

import click

import json
import pathlib
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import PIL.Image
from PIL import ImageFile
from PIL import UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
import dbimutils
import tqdm
from library import distortion


MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# CONFIGURE
CHARACTER_THRESHOLD = 0.85
GENERAL_THRESHOLD = 0.35


def load_labels() -> tuple[list[str], list[np.int64], list[np.int64], list[np.int64]]:
    path = huggingface_hub.hf_hub_download(CONV2_MODEL_REPO, LABEL_FILENAME)
    df = pd.read_csv(path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(model_repo, model_filename)
    model = rt.InferenceSession(
        path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return model

def new_square(image: PIL.Image.Image,height:int):
    image = image.convert("RGBA")

    old_size = image.size[0] if image.size[0] > image.size[1] else image.size[1]
    desired_size = max(old_size, height)
    
    new_image = PIL.Image.new("RGBA", (desired_size, desired_size), "WHITE")
    new_image.paste(image, ((desired_size - image.size[0]) // 2, (desired_size - image.size[1]) // 2))
    image = new_image.convert("RGB")
    return image

def predict(
    image: PIL.Image.Image,
    model,
    general_threshold: float,
    character_threshold: float,
    tag_names: list[str],
    rating_indexes: list[np.int64],
    general_indexes: list[np.int64],
    character_indexes: list[np.int64],
):

    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = image.convert("RGBA")
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image_test = image.copy()
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = dbimutils.make_square(image, height)
    print("squared")
    PIL.Image.fromarray(image).show()
    new_square(image_test, height).show()
    image = dbimutils.smart_resize(image, height)
    #print("resized")
    #PIL.Image.fromarray(image).show()
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick any where prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)
    # Everything else is characters: pick any where prediction confidence > threshold
    
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)

    return (character_res, rating, general_res)


def prepare(model_pair):
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels()
    tag_model = load_model(model_pair[0], model_pair[1])
    return tag_model, (tag_names, rating_indexes, general_indexes, character_indexes)


def invoke(
    path: pathlib.Path,
    model,
    labels,
    c_thresh,
    g_thresh,
    raw_values=False,
):

    if (
        path.suffix.endswith(".jpg")
        or path.suffix.endswith(".png")
        or path.suffix.endswith(".jpeg")
    ):
        im = PIL.Image.open(path)
        chara, rating, general_list = predict(
            im, model, g_thresh, c_thresh, *labels
        )
        im.close()
        real_rating = max(rating.items(), key=lambda x: x[1])[0]
        if raw_values:

            chara_list = [
                f"{char}: {round(c_thresh, 4)}" for char, c_thresh in chara.items()
            ]
            general_list = [
                f"{general}: {round(g_thresh, 4)}"
                for general, g_thresh in general_list.items()
            ]
        else:
            chara_list = [f"{char}" for char, _ in chara.items()]
            general_list = [f"{general}" for general, _ in general_list.items()]

        return general_list, chara_list, real_rating
    else:
        return [], [], ""


@click.group()
def karakar_grp():
    pass


@karakar_grp.command()
@click.argument("file_path")
@click.option("--model", default="SWIN")
@click.option("--character_threshold", "-c", default=CHARACTER_THRESHOLD, type=float)
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
def check(
    file_path,
    model: str,
    character_threshold: float,
    general_threshold: float,
):
    """Asks what the model thinks of the image."""

    file_path = pathlib.Path(file_path).resolve()
    if not file_path.exists() or not file_path.is_file():
        raise Exception("Either path is not a file or doesn't exist.")
    if model == "SWIN":
        model = (SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model == "CONV":
        model = (CONV_MODEL_REPO, MODEL_FILENAME)
    elif model == "CONV2":
        model = (CONV2_MODEL_REPO, MODEL_FILENAME)
    elif model == "VIT":
        model = (VIT_MODEL_REPO, MODEL_FILENAME)
    else:
        raise Exception(f"Unknown model: {model}")
    model_session, labels = prepare(model)
    general, characters, ratings = invoke(
        file_path,
        model_session,
        labels,
        character_threshold,
        general_threshold
    )
    print(f"> {file_path.name}")
    print(f"G: {', '.join(general)}")
    print(f"C: {', '.join(characters)}")
    print(f"R: {ratings}")


def sigmoid(x, k=1.0, a=2, b=-14, o=2):
    return round((k / (1.0 + np.exp(a + b * x))) + o, ndigits=2)


@karakar_grp.command()
@click.argument("folder")
@click.option("--model", default="SWIN")
@click.option("--character_threshold", "-c", default=CHARACTER_THRESHOLD, type=float)
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
@click.option("--reorder", is_flag=True, type=bool)
@click.option("--skip_exist", is_flag=True, type=bool)
def filter(
    folder,
    model: str,
    character_threshold: float,
    general_threshold: float,
    reorder: bool,
    skip_exist: bool,
):
    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise Exception("Either path is not a file or not a folder.")
    print("Folder Exists")
    print(f"Using thresholds {general_threshold} & {character_threshold}")
    if reorder:
        print("Filtering NSFW into folders")
        safe_folder = folder_path / "safe_media"
        safe_folder.mkdir(exist_ok=True)
        nsfw_folder = folder_path / "nsfw_media"
        nsfw_folder.mkdir(exist_ok=True)

    if model == "SWIN":
        model = (SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model == "CONV":
        model = (CONV_MODEL_REPO, MODEL_FILENAME)
    elif model == "VIT":
        model = (VIT_MODEL_REPO, MODEL_FILENAME)
    print("Model KP:", model)
    model_session, labels = prepare(model)
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
            if skip_exist and file.with_suffix(file.suffix + ".json").exists():
                continue
            try:
                general, characters, ratings = invoke(
                    file, model_session, labels, character_threshold, general_threshold
                )
            except UnidentifiedImageError:
                print(f"{file} is unidentified.")
                continue
            if ratings == "questionable" or ratings == "explicit":
                nsfw = ["nsfw"]
            else:
                nsfw = []

            with open(
                file.with_suffix(file.suffix + ".json"), "w", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "tags": general + nsfw,
                            "charas": characters,
                            "rating": ratings,
                        }
                    )
                )

            if reorder:
                tex = list(file.parent.glob(f"{file.stem}*.txt")) + list(
                    file.parent.glob(f"{file.stem}*.json")
                )
                if nsfw:
                    fi_file = nsfw_folder.joinpath(file.name)
                    file.rename(fi_file)
                    for tex_file in tex:
                        nsfw_folder.joinpath(tex_file.name)
                else:
                    file.rename(fi_file)
                    for tex_file in tex:
                        nsfw_folder.joinpath(tex_file.name)
    print("All done!")


if __name__ == "__main__":
    karakar_grp()

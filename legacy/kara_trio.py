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


SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# CONFIGURE
CHARACTER_THRESHOLD = 0.85
GENERAL_THRESHOLD = 0.35


def load_labels(model_repo) -> tuple[list[str], list[np.int64], list[np.int64], list[np.int64]]:
    path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME)
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


def predict(
    image: PIL.Image.Image,
    model,
    general_threshold: float,
    character_threshold: float,
    optimize: bool,
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
    image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    image = image[:, :, ::-1]

    image = dbimutils.make_square(image, height)
    image = dbimutils.smart_resize(image, height)
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
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels(model_pair[0])
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
            im, model, g_thresh, c_thresh, False, *labels
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
@click.argument("filename")
@click.option("--model", default="SWIN")
@click.option("--character_threshold", "-c", default=CHARACTER_THRESHOLD, type=float)
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
def check(
    filename,
    model: str,
    character_threshold: float,
    general_threshold: float,
):
    filename = pathlib.Path(filename).resolve()

    model_pairs = list([
        (SWIN_MODEL_REPO, MODEL_FILENAME),
        # (CONV_MODEL_REPO, MODEL_FILENAME),
        # (VIT_MODEL_REPO, MODEL_FILENAME),
        # Removed vit as it has a higher chance to produce false positives.
    ])

    json_file = filename.with_suffix(filename.suffix + ".json")
    if json_file.exists():
        meta = json.loads(json_file.read_text(encoding="utf-8"))
    else:
        meta = {}

    for model in model_pairs:
        print("Model KP:", model)
        model_session, labels = prepare(model)
        
        try:
            general, characters, rating = invoke(
                filename, model_session, labels, character_threshold, general_threshold
            )
        except UnidentifiedImageError:
            print(f"{filename} is unidentified.")
            continue
        nsfw = False if "nsfw" in meta.get("tags", []) else True
        nsfw = True if rating in ["questionable", "explicit"] else False
        print(set(general), character_threshold, general_threshold)
        new_tags = set(meta.get("tags", [])).union(set(general))
        new_charas = set(meta.get("charas", [])).union(set(characters))

        ndiff = set(general) - set(meta.get("tags", []))
        print(f"ndiff: {ndiff}")
        new_rating = meta["rating"] if "rating" in meta else rating
        meta.update(
            {
                "tags": list(new_tags),
                "charas": list(new_charas),
                "rating": new_rating,
            }
        )
    print(meta)

def set_composite(old_tags: set, new_tags: set, discard=False):
    count_tags = {}
    final_tags = set()
    for tag in old_tags:
        if tag[0] in ['1','2','3','4','5','6']:
            # count tag
            if tag[1] == "+":
                type_tag = tag[2:]
                count = 7
            else:
                type_tag = tag[1:]
                count = int(tag[0])
            if type_tag[-1] == 's':
                type_tag = type_tag[:-1]
            count_tags[type_tag] = count
        final_tags.add(tag)
    for tag in new_tags:
        if tag[0] in ['1','2','3','4','5','6']:
            # count tag
            if tag[1] == "+":
                type_tag = tag[2:]
                count = 7
            else:
                type_tag = tag[1:]
                count = int(tag[0])
            if type_tag[-1] == 's':
                type_tag = type_tag[:-1]
            if type_tag in count_tags and count_tags[type_tag] != count:
                if discard:
                    print(f"{type_tag} found in inital set, discard option enabled.")
                    continue
                else:
                    print(f"{type_tag} found in inital set, discard option disabled.")
        final_tags.add(tag)
    return final_tags

@karakar_grp.command()
@click.argument("folder")
@click.option("--model", default="SWIN")
@click.option("--character_threshold", "-c", default=CHARACTER_THRESHOLD, type=float)
@click.option("--general_threshold", "-g", default=GENERAL_THRESHOLD, type=float)
@click.option("--discard_conflict", "-d", is_flag=True, default=False)
def filter(
    folder,
    model: str,
    character_threshold: float,
    general_threshold: float,
    discard_conflict: bool
):

    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise Exception("Either path is not a file or not a folder.")
    print("Folder Exists")
    model_pairs = [
        (SWIN_MODEL_REPO, MODEL_FILENAME),
        (CONV_MODEL_REPO, MODEL_FILENAME),
        # (VIT_MODEL_REPO, MODEL_FILENAME),
        # Removed vit as it has a higher chance to produce false positives.
    ]
    model_name = [
        "SwinV2",
        "ConvNeXtV2"
    ]
    for idx, model in enumerate(model_pairs):
        print("Model KP:", model)
        model_session, labels = prepare(model)
        print("Model loaded. Counting...")
        ctr = 0
        for _ in distortion.folder_images(folder_path):
            ctr += 1
        print("Found", ctr, "images.")
        with tqdm.tqdm(
            desc="Tagging: ?", dynamic_ncols=True, unit="file", total=ctr
        ) as pbar:
            for file in distortion.folder_images(folder_path):
                pbar.desc = f"Tagging: {file.name}"
                pbar.update(1)
                if file.stat().st_size == 0:
                    continue

                json_file = file.with_suffix(file.suffix + ".json")
                if json_file.exists():
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                else:
                    meta = {}
                try:
                    general, characters, rating = invoke(
                        file, model_session, labels, character_threshold, general_threshold
                    )
                except UnidentifiedImageError:
                    print(f"{file} is unidentified.")
                    continue
                nsfw = False if "nsfw" in meta.get("tags", []) else True
                nsfw = True if rating in ["questionable", "explicit"] else False
                new_rating = meta["rating"] if "rating" in meta else rating
                tags = {**meta.get("tags",{}),
                        f"{model_name[idx]}": list(general)
                }
                chars = {**meta.get("chars",{}),
                         f"{model_name[idx]}": list(characters)
                }
                meta.update(
                    {
                        "tags": tags,
                        "chars": chars,
                        "rating": new_rating,
                    }
                )
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
    print("All done!")


if __name__ == "__main__":
    karakar_grp()

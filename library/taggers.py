import typing

import huggingface_hub
import numpy
import pandas
from PIL import Image
from torch import Tensor
from .taggers_core import OnnxLoader, HuggingLoader

hugging_pipelines = {
    "cafe_style": ["image-classification", "cafeai/cafe_style"],
    "cafe_aesthetic": ["image-classification", "cafeai/cafe_aesthetic"],
}

onnx_pipeline = {
    "skytnt_aesthetic": ["skytnt/anime-aesthetic", "model.onnx"],
    # "skytnt_deepdanbooru": ["skytnt/deepdanbooru_onnx", "deepdanbooru.onnx"],
    # WD 1.4
    "wd_moat": [
        "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "model.onnx",
        "selected_tags.csv",
    ],
    "wd_swinv2": [
        "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
        "model.onnx",
        "selected_tags.csv",
    ],
    "wd_convnext": [
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "model.onnx",
        "selected_tags.csv",
    ],
    "wd_convnextv2": [
        "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        "model.onnx",
        "selected_tags.csv",
    ],
    "wd_vit": ["SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx", "selected_tags.csv"],
}


class SkyTNTTagger(OnnxLoader):
    def __init__(self, model) -> None:
        if not model.startswith("skytnt_"):
            raise Exception(
                f"Expected model with \"skytnt_*\". [{', '.join(list(onnx_pipeline.keys()))}]"
            )
        self.load_model()

    def predict(self, image: Image.Image):
        batch, size, w, h = self.model.get_inputs()[0].shape
        np = self.image2numpy(image, w, bg_color=[0, 0, 0])
        print(np.shape)


class SkyTNTAesthetic(SkyTNTTagger):
    def __init__(self) -> None:
        super().__init__("skytnt_aesthetic")

    def predict(self, image: Image.Image):
        return super().predict(image)


class CafeTagger(HuggingLoader):
    def __init__(self, model) -> None:
        if not model.startswith("cafe_"):
            raise Exception(
                f"Expected model with \"cafe_*\". [{', '.join(list(hugging_pipelines.keys()))}]"
            )

        super().__init__(hugging_pipelines.get(model, []))

    def predict(self, image: Image.Image, scale: float = 100.0):

        if self.pipeline is None:
            raise Exception("Pipeline is not ready!")
        predictions = self.pipeline(image, top_k=2)
        predict_keyed = {}
        if predictions is None:
            raise Exception("Predictions missing?")
        for p in predictions:
            if not isinstance(p, Tensor):
                raise Exception("Prediction value is missing?")
            predict_keyed[p["label"]] = p["score"] * scale
        return predict_keyed

    def predict_generator(self, generator: typing.Generator, scale: float = 100.0):
        if self.pipeline is None:
            raise Exception("Pipeline is not ready!")
        for predictions in self.pipeline(generator, top_k=2):
            predict_keyed = {}
            for p in predictions:
                predict_keyed[p["label"]] = p["score"] * scale
            yield predict_keyed


class CafeAesthetic(CafeTagger):
    def __init__(self) -> None:
        super().__init__("cafe_aesthetic")

    def predict(self, image: Image.Image, scale: float = 100):
        predict_keys = super().predict(image, scale)
        return round(predict_keys["aesthetic"], 2)


class CafeWaifu(CafeTagger):
    def __init__(self) -> None:
        super().__init__("cafe_waifu")

    def predict(self, image: Image.Image, scale: float = 100):
        predict_keys = super().predict(image, scale)
        return round(predict_keys["waifu"], 2)


class WDTagger(OnnxLoader):

    MODELS = [i.split("_")[-1] for i in onnx_pipeline.keys() if i.startswith("wd_")]

    def load_labels(self):
        if not self.labels:
            path = huggingface_hub.hf_hub_download(
                self.model_mapping[0], self.model_mapping[2]
            )
            datafile = pandas.read_csv(path)
            tag_names = datafile["name"].tolist()
            rating_indexes = list(numpy.where(datafile["category"] == 9)[0])
            general_indexes = list(numpy.where(datafile["category"] == 0)[0])
            character_indexes = list(numpy.where(datafile["category"] == 4)[0])
            self.labels = [
                tag_names,
                rating_indexes,
                general_indexes,
                character_indexes,
            ]
        return self.labels

    def __init__(self, model="wd_swinv2") -> None:
        if not model.startswith("wd_"):
            raise Exception(
                f"Expected model with \"wd_*\". [{', '.join(list(onnx_pipeline.keys()))}]"
            )
        super(WDTagger, self).__init__(onnx_pipeline.get(model, []))
        self.labels = []
        self.load_labels()
        self.load_model()

    def predict(
        self,
        image: Image.Image,
        general_threshold: float,
        character_threshold: float,
    ):

        _, height, _, _ = self.model.get_inputs()[0].shape

        np_img = self.image2numpy(image, height)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: np_img})[0]

        labels = list(zip(self.labels[0], probs[0].astype(float)))

        ratings_names = [labels[i] for i in self.labels[1]]
        rating = dict(ratings_names)

        general_names = [labels[i] for i in self.labels[2]]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)
        # Everything else is characters: pick any where prediction confidence > threshold

        character_names = [labels[i] for i in self.labels[3]]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        return (character_res, rating, general_res)

import numpy
import onnxruntime
import huggingface_hub, pandas
import cv2
from PIL import Image

hugging_pipelines = {
    "cafe_style": ["image-classification", "cafeai/cafe_style"],
    "cafe_aesthetic": ["image-classification", "cafeai/cafe_aesthetic"],
}

onnx_pipeline = {
    "skytnt_aesthetic": ["skytnt/anime-aesthetic", "model.onnx"],
    # "skytnt_deepdanbooru": ["skytnt/deepdanbooru_onnx", "deepdanbooru.onnx"],
    # WD 1.4
    "wd_moat": ["SmilingWolf/wd-v1-4-moat-tagger-v2", "model.onnx"],
    "wd_swinv2": ["SmilingWolf/wd-v1-4-swinv2-tagger-v2", "model.onnx"],
    "wd_convnext": ["SmilingWolf/wd-v1-4-convnext-tagger-v2", "model.onnx"],
    "wd_convnextv2": ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "model.onnx"],
    "wd_vit": ["SmilingWolf/wd-v1-4-vit-tagger-v2", "model.onnx"],
} 

class OnnxLoader:

    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, mapping) -> None:
        self.model_mapping = mapping

    def load_model(self):
        path = huggingface_hub.hf_hub_download(
            self.model_mapping[0], self.model_mapping[1]
        )
        model = onnxruntime.InferenceSession(path, providers=self.PROVIDERS)
        self.model = model

class HuggingLoader:

    def __init__(self, mapping) -> None:
        self.model_mapping = mapping

    def load_model(self):
        pass


class SkyTNTAesthetic(OnnxLoader):
    def __init__(self, model="skytnt_aesthetic") -> None:
        if not model.startswith("skytnt_"):
            raise Exception(
                f"Expected model with \"wd_*\". [{', '.join(list(onnx_pipeline.keys()))}]"
            )
        self.load_model()

class WDTagger(OnnxLoader):

    MODELS = [i.split("_")[-1] for i in onnx_pipeline.keys() if i.startswith("wd_")]

    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def load_labels(self):
        if not self.labels:
            path = huggingface_hub.hf_hub_download(
                self.model_mapping[0], "selected_tags.csv"
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
        self.model = None
        self.labels = []
        self.load_labels()
        self.load_model()

    def image2numpy(self, image, height):
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image_test = image.copy()
        image = numpy.asarray(image)

        image = image[:, :, ::-1]

        old_size = image.shape[:2]
        desired_size = max(old_size)
        desired_size = max(desired_size, height)

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [255, 255, 255]
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        if image.shape[0] > height:
            image = cv2.resize(image, (height, height), interpolation=cv2.INTER_AREA)
        elif image.shape[0] < height:
            image = cv2.resize(image, (height, height), interpolation=cv2.INTER_CUBIC)
        image = image.astype(numpy.float32)
        image = numpy.expand_dims(image, 0)
        return image

    def predict(
        self,
        image: Image.Image,
        general_threshold: float,
        character_threshold: float,
    ):
        
        _, height, _, _ = self.model.get_inputs()[0].shape

        np_img = self.image2numpy(image,height)

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

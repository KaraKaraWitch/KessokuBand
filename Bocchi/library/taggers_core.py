import typing

import cv2
import huggingface_hub
import numpy
import onnxruntime
from PIL import Image
from transformers import pipeline

class OnnxLoader:

    PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, mapping) -> None:
        self.model_mapping = mapping
        self.model: onnxruntime.InferenceSession

    def load_model(self):
        path = huggingface_hub.hf_hub_download(
            self.model_mapping[0], self.model_mapping[1]
        )
        model = onnxruntime.InferenceSession(path, providers=self.PROVIDERS)
        self.model = model

    def image2numpy(
        self,
        image_raw: Image.Image,
        size: int,
        bg_color: typing.Optional[list[int]] = None,
        cast_type=numpy.float32,
    ):
        """Converts images into a 2d numpy image

        Args:
            image_raw (Image.Image): The raw image to be converted
            size (int): The Final size for the image to be resized to
            bg_color (typing.Optional[list[int]], optional): An optional background color. Defaults to None/White.

        Returns:
            _type_: an
        """
        rgba_image = image_raw.convert("RGBA")
        new_image = Image.new("RGBA", rgba_image.size, "WHITE")
        new_image.paste(rgba_image, mask=rgba_image)
        image = new_image.convert("RGB")
        image = numpy.asarray(image)

        image = image[:, :, ::-1]

        old_size = image.shape[:2]
        desired_size = max(old_size)
        desired_size = max(desired_size, size)

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        if not bg_color:
            color = [255, 255, 255]
        else:
            color = bg_color
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        if image.shape[0] > size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        elif image.shape[0] < size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        image = image.astype(cast_type)
        image = numpy.expand_dims(image, 0)
        return image
    
    def predict(self,**kwargs):
        raise NotImplementedError()


class HuggingLoader:
    def __init__(self, mapping) -> None:
        self.model_mapping = mapping
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if self.pipeline:
            return
        self.pipeline = pipeline(
            self.model_mapping[0], model=self.model_mapping[1], device=0
        )

    def predict(self,**kwargs):
        raise NotImplementedError()
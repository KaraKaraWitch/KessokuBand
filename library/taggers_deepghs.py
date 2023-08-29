import typing

import huggingface_hub
import numpy
from PIL import Image
from .taggers_core import OnnxLoader

try:
    import orjson as json
except ImportError:
    print(
        "[KessokuTaggers] orjson not installed. Consider installing for better performance."
    )
    import json


onnx_pipeline = {
    # GHS Styles
    "ghs_style": [
        "deepghs/anime_classification",
        "caformer_s36_plus/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_eye": [
        "deepghs/anime_ch_eye_color",
        "caformer_s36_raw/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_skin": [
        "deepghs/anime_ch_skin_color",
        "caformer_s36/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_horn": [
        "deepghs/anime_ch_horn",
        "caformer_s36_raw/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_ear": [
        "deepghs/anime_ch_ear",
        "caformer_s36_raw/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_hair_color": [
        "deepghs/anime_ch_hair_color",
        "caformer_s36_v0/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghs_hair_length": [
        "deepghs/anime_ch_hair_length",
        "caformer_s36_v0/model.onnx",
        "caformer_s36_plus/meta.json",
    ],
    "ghb_face": [
        "deepghs/anime_face_detection",
        "face_detect_v1.4_s/model.onnx",
    ],
    "ghb_head": [
        "deepghs/anime_head_detection",
        "head_detect_v0_s/model.onnx",
    ],
    "ghb_person": [
        "deepghs/anime_person_detection",
        "person_detect_v1.3_s/model.onnx",
    ],
    "ghb_censor": [
        "deepghs/anime_censor_detection",
        "censor_detect_v1.0_s/model.onnx",
    ]
}



class DeepGHSTagger(OnnxLoader):

    MODELS = [i.split("_")[-1] for i in onnx_pipeline.keys() if i.startswith("ghs_")]

    def load_labels(self):
        if not self.labels:
            path = huggingface_hub.hf_hub_download(
                self.model_mapping[0], self.model_mapping[2]
            )

        return self.labels

    _DEFAULT_ORDER = "HWC"

    def get_numpy_order(self, order: str):
        """Gets the order that the numpy array should be.

        NOTE: This is pulled from imgutils

        Args:
            order (str): The order for the numpy.

        Returns:
            str: Return the numpy map.
        """
        return tuple(self._DEFAULT_ORDER.index(c) for c in order.upper())

    def image2numpy(
        self,
        image: Image.Image,
        size: typing.Tuple[int, int] = (384, 384),
        normalize: typing.Optional[typing.Tuple[float, float]] = (0.5, 0.5),
        order: str = "CHW",
        to_float: bool = True,
    ):
        """Converts images into a 2d numpy image

        NOTE: This is pulled from imgutils

        Args:
            image_raw (Image.Image): The raw image to be converted
            size (int): The Final size for the image to be resized to
            bg_color (typing.Optional[list[int]], optional): An optional background color. Defaults to None/White.

        Returns:
            _type_: an
        """
        image = image.resize(size, Image.BILINEAR)
        array = numpy.asarray(image)
        array = numpy.transpose(array, self.get_numpy_order(order))
        if to_float:
            array = (array / 255.0).astype(numpy.float32)
        if normalize:
            mean, std = normalize
            mean = numpy.asarray([mean]).reshape((-1, 1, 1))
            std = numpy.asarray([std]).reshape((-1, 1, 1))
            array = (array - mean) / std
        return array

    def __init__(self, model="ghs_") -> None:
        if not model.startswith("ghs_"):
            raise Exception(
                f"Expected model with \"ghs_*\". [{', '.join(list(onnx_pipeline.keys()))}]"
            )
        super(DeepGHSTagger, self).__init__(onnx_pipeline.get(model, []))
        # self.model = None
        self.labels = []
        self.load_labels()
        self.load_model()

    # def predict_object(self, image: Image.Image, iou_thr:float):

    def predict(self, image: Image.Image, threshold:float):
        pass

class DeepGHSObjectTagger(OnnxLoader):

    MODELS = [i.split("_")[-1] for i in onnx_pipeline.keys() if i.startswith("ghb_")]

    @staticmethod
    def _yolo_xywh2xyxy(x: numpy.ndarray) -> numpy.ndarray:
        """
        Copied from yolov8.

        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = numpy.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    @staticmethod
    def _yolo_nms(boxes, scores, thresh: float = 0.7) -> typing.List[int]:
        """
        dets: ndarray, (num_boxes, 5)
            每一行表示一个bounding box：[xmin, ymin, xmax, ymax, score]
            其中xmin, ymin, xmax, ymax分别表示框的左上角和右下角坐标，score表示框的分数
        thresh: float
            两个框的IoU阈值
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按照score降序排列
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算其他所有框与当前框的IoU
            xx1 = numpy.maximum(x1[i], x1[order[1:]])
            yy1 = numpy.maximum(y1[i], y1[order[1:]])
            xx2 = numpy.minimum(x2[i], x2[order[1:]])
            yy2 = numpy.minimum(y2[i], y2[order[1:]])

            w = numpy.maximum(0.0, xx2 - xx1 + 1)
            h = numpy.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU小于阈值的框
            inds = numpy.where(iou <= thresh)[0]
            order = order[inds + 1]

        return keep

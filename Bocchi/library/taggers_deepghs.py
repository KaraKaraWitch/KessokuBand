import math
import typing

import huggingface_hub
import numpy
from PIL import Image, ImageDraw, ImageFont
from .taggers_core import OnnxLoader

try:
    import orjson as json
except ImportError:
    print(
        "[KessokuTaggers] orjson not installed. Consider installing for improved deserialization performance."
    )

try:
    import lsnms
except ImportError:
    lsnms = None

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
    # Object detection
    # S: Small, N: Nano, M: Medium
    "ghb_face": [
        "deepghs/anime_face_detection",
        "face_detect_v1.4_s/model.onnx",
    ],
    "ghb_head": [
        "deepghs/imgutils-models",
        "head_detect/head_detect_best_s.onnx",
    ],
    "ghb_person": [
        "deepghs/imgutils-models",
        "person_detect/person_detect_plus_v1.1_best_m.onnx",
    ],
    "ghb_censor": [
        "deepghs/anime_censor_detection",
        "censor_detect_v1.0_s/model.onnx",
    ],
    "ghb_half": [
        "deepghs/anime_halfbody_detection",
        "halfbody_detect_v0.4_s/model.onnx",
    ]
}


class DeepGHSCore(OnnxLoader):

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
        # size: typing.Tuple[int, int] = (384, 384),
        normalize: typing.Optional[typing.Tuple[float, float]] = (0.5, 0.5),
        order: str = "CHW",
        to_float: bool = True,
    ):
        """Converts images into a 2d numpy image

        NOTE: This is pulled from imgutils, also known as rgb_encode

        Args:
            image_raw (Image.Image): The raw image to be converted
            size (int): The Final size for the image to be resized to
            bg_color (typing.Optional[list[int]], optional): An optional background color. Defaults to None/White.

        Returns:
            _type_: an
        """
        # image = image.resize(size, Image.BILINEAR)
        array = numpy.asarray(image)
        array = numpy.transpose(array, self.get_numpy_order(order))
        if to_float:
            array = (array / 255.0).astype(numpy.float32)
            # print(array.dtype)
        if normalize:
            flt_type = numpy.float32 if to_float else None
            mean, std = normalize
            mean = numpy.asarray([mean], dtype=flt_type).reshape((-1, 1, 1))
            std = numpy.asarray([std], dtype=flt_type).reshape((-1, 1, 1))
            array = (array - mean) / std
        return array

    def detection_visualize(self, image: Image.Image, detection: typing.List[typing.Tuple[typing.Tuple[float, float, float, float], str, float]],
                        labels: typing.Optional[typing.List[str]] = None, text_padding: int = 6, fontsize: int = 12,
                        no_label: bool = False):
        """
        Overview:
            Visualize the results of the object detection.
        :param image: Image be detected.
        :param detection: The detection results list, each item includes the detected area `(x0, y0, x1, y1)`,
            the target type (always `head`) and the target confidence score.
        :param labels: An array of known labels. If not provided, the labels will be automatically detected
            from the given ``detection``.
        :param text_padding: Text padding of the labels. Default is ``6``.
        :param fontsize: Font size of the labels. At runtime, an attempt will be made to retrieve the font used
            for rendering from `matplotlib`. Therefore, if `matplotlib` is not installed, only the default pixel font
            provided with `Pillow` can be used, and the font size cannot be changed.
        :param no_label: Do not show labels. Default is ``False``.
        :return: A `PIL` image with the same size as the provided image `image`, which contains the original image
            content as well as the visualized bounding boxes.
        Examples::
            See :func:`imgutils.detect.face.detect_faces` and :func:`imgutils.detect.person.detect_person` for examples.
        """
        # image = load_image(image, force_background=None, mode='RGBA')
        visual_image = image.copy()
        draw = ImageDraw.ImageDraw(visual_image, mode='RGBA')
        font = ImageFont.load_default()

        labels = sorted(labels or {label for _, label, _ in detection})
        # _colors = list(map(str, rnd_colors(len(labels))))
        # _color_map = dict(zip(labels, _colors))
        for (xmin, ymin, xmax, ymax), label, score in detection:
            box_color = "#000000"
            draw.rectangle((xmin, ymin, xmax, ymax), outline=box_color, width=2)

            if not no_label:
                label_text = f'{label}: {score * 100:.2f}%'
                _t_x0, _t_y0, _t_x1, _t_y1 = draw.textbbox((xmin, ymin), label_text, font=font)
                _t_width, _t_height = _t_x1 - _t_x0, _t_y1 - _t_y0
                if ymin - _t_height - text_padding < 0:
                    _t_text_rect = (xmin, ymin, xmin + _t_width + text_padding * 2, ymin + _t_height + text_padding * 2)
                    _t_text_co = (xmin + text_padding, ymin + text_padding)
                else:
                    _t_text_rect = (xmin, ymin - _t_height - text_padding * 2, xmin + _t_width + text_padding * 2, ymin)
                    _t_text_co = (xmin + text_padding, ymin - _t_height - text_padding)

                draw.rectangle(_t_text_rect, fill=f"{box_color}80")
                draw.text(_t_text_co, label_text, fill="black", font=font)

        return visual_image

class DeepGHSTagger(DeepGHSCore):

    MODELS = ["_".join(i.split("_")[1:]) for i in onnx_pipeline.keys() if i.startswith("ghs_")]

    def load_labels(self):
        if not self.labels:
            path = huggingface_hub.hf_hub_download(
                self.model_mapping[0], self.model_mapping[2]
            )

        return self.labels

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

class DeepGHSObjectTagger(DeepGHSCore):

    MODELS = [i.split("_")[-1] for i in onnx_pipeline.keys() if i.startswith("ghb_")]

    def unpack_yolo_xywh(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Copied from yolov8.

        NOTE: Taken from https://huggingface.co/spaces/deepghs/anime_object_detection/blob/main/yolo_.py

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


    def __init__(self, model="ghb_") -> None:
        if not model.startswith("ghb_"):
            raise Exception(
                f"Expected model with \"ghs_*\". [{', '.join(list(onnx_pipeline.keys()))}]"
            )
        super(DeepGHSObjectTagger, self).__init__(onnx_pipeline.get(model, []))
        # self.model = None
        # self.labels = []
        # self.load_labels()
        self.load_model()

    def non_max_supp(self, boxes, scores, thresh: float = 0.7, try_accel=False) -> typing.List[int]:
        """

        Non Maximal Suppression: 

        NOTE: Taken from https://huggingface.co/spaces/deepghs/anime_object_detection/blob/main/yolo_.py

        dets: ndarray, (num_boxes, 5)
            每一行表示一个bounding box：[xmin, ymin, xmax, ymax, score]
            其中xmin, ymin, xmax, ymax分别表示框的左上角和右下角坐标，score表示框的分数
        thresh: float
            两个框的IoU阈值
        """

        if try_accel and lsnms is not None:
            return lsnms.nms(boxes,scores, iou_threshold=thresh)

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

    def _image_preprocess(self, image: Image.Image, max_infer_size: int = 640, align: int = 32):
        old_width, old_height = image.width, image.height
        new_width, new_height = old_width, old_height
        r = max_infer_size / max(new_width, new_height)
        if r < 1:
            new_width, new_height = new_width * r, new_height * r
        new_width = int(math.ceil(new_width / align) * align)
        new_height = int(math.ceil(new_height / align) * align)
        image = image.resize((new_width, new_height))
        return image, (old_width, old_height), (new_width, new_height)

    def unpack_box(self, x, y, old_size, new_size):
        old_width, old_height = old_size
        new_width, new_height = new_size
        x, y = x / new_width * old_width, y / new_height * old_height
        x = int(numpy.clip(x, a_min=0, a_max=old_width).round())
        y = int(numpy.clip(y, a_min=0, a_max=old_height).round())
        return x, y

    def _yolo_postprocess(self, output, conf_threshold, iou_threshold, old_size, new_size, labels: typing.List[str], try_accel=False):
        max_scores = output[4:, :].max(axis=0)
        output = output[:, max_scores > conf_threshold].transpose(1, 0)
        boxes = output[:, :4]
        scores = output[:, 4:]
        filtered_max_scores = scores.max(axis=1)

        if not boxes.size:
            return []

        boxes = self.unpack_yolo_xywh(boxes)
        idx = self.non_max_supp(boxes, filtered_max_scores, thresh=iou_threshold, try_accel=try_accel)
        boxes, scores = boxes[idx], scores[idx]

        detections = []
        for box, score in zip(boxes, scores):
            x0, y0 = self.unpack_box(box[0], box[1], old_size, new_size)
            x1, y1 = self.unpack_box(box[2], box[3], old_size, new_size)
            max_score_id = score.argmax()
            detections.append(((x0, y0, x1, y1), labels[max_score_id], float(score[max_score_id])))

        return detections

    def predict(self, image: Image.Image, conf_threshold:float=0.3, iou_threshold=0.5, infer_size: int=640, preview:bool=False):
        # dt = time.monotonic()
        new_image, old_size, new_size = self._image_preprocess(image, infer_size)
        # print(time.monotonic() - dt, "preprocess seconds")
        # dt = time.monotonic()
        array = self.image2numpy(new_image)
        # print(time.monotonic() - dt, "image2numpy seconds")
        # dt = time.monotonic()
        output, = self.model.run(['output0'], {'images':[array]})
        # print(time.monotonic() - dt, "model output")
        if not preview:
            return self._yolo_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['person'])
        if preview:
            detections = self._yolo_postprocess(output[0], conf_threshold, iou_threshold, old_size, new_size, ['person'])
            self.detection_visualize(image, detections).show()
            return detections
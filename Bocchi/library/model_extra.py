import typing

import pydantic


class BoundBox(pydantic.BaseModel):
    bounds: list
    confidence: float
    # top_right: int


class NullBoundBox(BoundBox):
    def __init__(self):
        self.bounds = [0, 0, 0, 0]
        self.confidence = 0.0


class Character(pydantic.BaseModel):
    person: BoundBox
    head: typing.Optional[typing.List[BoundBox]] = []
    eye_color: typing.Optional[str] = ""
    skin_color: typing.Optional[str] = ""
    horns: typing.Optional[str] = ""
    ears: typing.Optional[str] = ""
    hair: typing.Optional[str] = ""


class ImageMetaAdditive(pydantic.BaseModel):
    style_type: typing.Optional[str] = ""
    persons: typing.List[Character] = []

    def to_dict(self):
        return self.model_dump(mode="json",exclude_unset=True)

    @classmethod
    def from_dict(cls, dict: dict):
        return cls.model_validate(dict)


def predictions_to_boundbox(prediction: list):
    boxes = []
    for pred in prediction:
        boundbox, _, confidence = pred
        boxes.append(BoundBox(bounds=list(boundbox), confidence=confidence))
    return boxes

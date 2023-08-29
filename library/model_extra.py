import enum
import typing

import pydantic

class Character(pydantic.BaseModel):

    head: typing.Optional[typing.List[float]] = []
    eye_color: typing.Optional[str] = ""
    skin_color: typing.Optional[str] = ""
    horns: typing.Optional[str] = ""
    ears: typing.Optional[str] = ""
    hair: typing.Optional[str] = ""

class ImageMetaAdditive(pydantic.BaseModel):

    style_type: typing.Optional[str] = ""
    persons: typing.Optional[typing.List[Character]] = []

    def to_dict(self):
        return self.dict(exclude_unset=True)
    
    @classmethod
    def from_dict(cls, dict:dict):
        return cls.parse_obj(dict)
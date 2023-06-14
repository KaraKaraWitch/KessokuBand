import typing
import pydantic
import enum

class Tags(pydantic.BaseModel):

    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []
    Booru: typing.Optional[typing.List[str]] = []

class Chars(pydantic.BaseModel):

    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []
    Booru: typing.Optional[typing.List[str]] = []

class TaggerMapping(enum.Enum):

    MOAT = "moat"
    SwinV2 = "swinv2"
    ConvNext = "convnext"
    ConvNextV2 = "convnextv2"
    ViT = "vit"
    Booru = "booru"

class SankakuExtra(pydantic.BaseModel):
    weeb_flags: list
    

class Extra(pydantic.BaseModel):
    size: typing.Optional[tuple] = ()
    source: typing.Optional[str] = ""
    uploader: typing.Optional[str] = ""
    weeb_flags: typing.Optional[typing.List[str]] = []
    artists: typing.Optional[typing.List[str]] = []

class Scoring(pydantic.BaseModel):
    Booru: typing.Optional[float] = None
    Cafe: typing.Optional[float] = None
    SkyTnt: typing.Optional[float] = None

class RatingEnum(str, enum.Enum):
    GENERAL = "general"
    SENSITIVE = "sensitive"
    QUESTIONABLE = "questionable"
    EXPLICIT = "explicit"

class ImageMeta(pydantic.BaseModel):

    tags: Tags
    chars: Chars
    rating: RatingEnum = RatingEnum.GENERAL
    extra: typing.Optional[Extra] = None
    score: typing.Optional[Scoring] = None

    def to_dict(self):
        return self.dict(exclude_unset=True)
    
    @classmethod
    def from_dict(cls, dict:dict):
        return cls.parse_obj(dict)
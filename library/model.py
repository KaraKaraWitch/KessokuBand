import typing
import pydantic
import enum

class Tags(pydantic.BaseModel):
    """Contains dataclass storing list of tags for the image.

    Args:
        pydantic (_type_): _description_
    """

    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []
    Booru: typing.Optional[typing.List[str]] = []

class Chars(pydantic.BaseModel):
    """Contains dataclass storing list of characters.
    """

    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []
    Booru: typing.Optional[typing.List[str]] = []

class TaggerMapping(enum.Enum):
    """Enum to map string to attributes within Tag/Chars"""

    MOAT = "moat"
    SwinV2 = "swinv2"
    ConvNext = "convnext"
    ConvNextV2 = "convnextv2"
    ViT = "vit"
    Booru = "booru"



class SankakuExtra(pydantic.BaseModel):
    """Some extra data from sankaku [Sankaku exclusive only]
    """
    weeb_flags: list
    

class Extra(pydantic.BaseModel):
    size: typing.Optional[tuple] = ()
    source: typing.Optional[str] = ""
    uploader: typing.Optional[str] = ""
    weeb_flags: typing.Optional[typing.List[str]] = []
    artists: typing.Optional[typing.List[str]] = []

class ScoringMapping(str, enum.Enum):
    Booru = "booru"
    CafeAesthetic = "cafe_aesthetic"
    SkyTntAesthetic = "skytnt_aesthetic"
    CafeWaifu = "cafe_waifu"
    CafeStyle = "cafe_style"


class Scoring(pydantic.BaseModel):
    """Score dataclass to contain various scoring functions
    """
    Booru: typing.Optional[float] = None
    CafeAesthetic: typing.Optional[float] = None
    SkyTntAesthetic: typing.Optional[float] = None
    CafeWaifu: typing.Optional[float] = None
    CafeStyle: typing.Optional[dict] = None
    

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
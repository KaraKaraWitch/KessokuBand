import enum
import typing

import pydantic


class Tags(pydantic.BaseModel):
    """Contains dataclass storing list of tags for the image.

    Args:
        pydantic (_type_): _description_
    """

    Booru: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    EVA2_3: typing.Optional[typing.List[str]] = []
    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2_3: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []

class Chars(pydantic.BaseModel):
    """Contains dataclass storing list of characters.
    """

    Booru: typing.Optional[typing.List[str]] = []
    ConvNext: typing.Optional[typing.List[str]] = []
    ConvNextV2: typing.Optional[typing.List[str]] = []
    EVA2_3: typing.Optional[typing.List[str]] = []
    MOAT: typing.Optional[typing.List[str]] = []
    SwinV2_3: typing.Optional[typing.List[str]] = []
    SwinV2: typing.Optional[typing.List[str]] = []
    ViT: typing.Optional[typing.List[str]] = []

class TaggerMapping(enum.Enum):
    """Enum to map string to attributes within Tag/Chars"""

    Booru = "booru"
    ConvNext = "convnext"
    ConvNextV2 = "convnextv2"
    EVA2_3="eva02_3"
    MOAT = "moat"
    SwinV2 = "swinv2"
    SwinV2_3 = "swinv2_3"
    ViT = "vit"

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
    ClipMLPAesthetic = "clip_aesthetic"
    CafeWaifu = "cafe_waifu"
    CafeStyle = "cafe_style"
    


class Scoring(pydantic.BaseModel):
    """Score dataclass to contain various scoring functions
    """
    Booru: typing.Optional[float] = None
    BooruSumLikes: typing.Optional[int] = None
    CafeAesthetic: typing.Optional[float] = None
    ClipMLPAesthetic: typing.Optional[float] = None
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


def extract_characters(general_tags:list[str]):
    characters = set([
        "1girl",
        "2girls",
        "3girls",
        "4girls",
        "5girls",
        "6+girls",
        "1boy",
        "2boys",
        "3boys",
        "4boys",
        "5boys",
        "6+boys",
        "1other",
        "2others",
        "3others",
        "4others",
        "5others",
        "6+others",
    ])
    
    extracted_general_tags = set(general_tags) - characters
    
    return list(characters.intersection(set(general_tags))), list(extracted_general_tags)
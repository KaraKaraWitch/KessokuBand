import pathlib
from . import model
from itertools import takewhile, zip_longest


def drum_fallback(tags_list: list[dict]):
    print("Drumroll fallback was used. suspected malformed data!")
    types = {
        0: "tags",  # general tags
        1: "artist",  # artist
        2: "studio",  # studio
        3: "copyright",  # copyright
        4: "character",  # character
        5: "tags",  # genre tags
        6: "tags_6",  # unknown
        7: "tags_7",  # unknown
        8: "tags",  # medium tags
        9: "tags",  # meta tags
    }
    d2 = {}
    for tag in tags_list:
        grp = types.get(tag["type"], f"tags_{tag['type']}")
        tag_group: set = d2.setdefault(grp, set())
        if tag["name_en"] in tag_group:
            continue
        tag_group.add(tag["name_en"])
        d2[grp] = tag_group
    for k, v in d2.items():
        if isinstance(v, set):
            d2[k] = list(v)
    return d2


def drumroll_resolver(sankaku_contents: dict, boccModel: model.ImageMeta):
    fb = {}
    if len(sankaku_contents.get("tags", [])) > 0 and isinstance(
        sankaku_contents["tags"][0], dict
    ):
        print(sankaku_contents["id"])
        fb = drum_fallback(sankaku_contents["tags"])
        sankaku_contents["tags"] = []
    sankaku_contents.update(fb)

    tags = sankaku_contents["tags"] + sankaku_contents.get("copyright", [])
    character = sankaku_contents.get("character", [])
    boccModel.tags.Booru = tags
    boccModel.chars.Booru = character
    rating = sankaku_contents["rating"].lower()
    if rating == "safe":
        rating = "general"
    boccModel.rating = model.RatingEnum(rating)
    if not boccModel.extra:
        boccModel.extra = model.Extra()
    size = tuple(sankaku_contents["size"])
    source = sankaku_contents.get("source")
    source = source if source else ""
    boccModel.extra.size = size
    boccModel.extra.source = source
    boccModel.extra.uploader = sankaku_contents["user"]
    boccModel.extra.artists = sankaku_contents.get("artist", []) + sankaku_contents.get(
        "studio", []
    )
    if not boccModel.score:
        boccModel.score = model.Scoring()
    boccModel.score.Booru = round(sankaku_contents["score"], 2)
    boccModel.score.BooruSumLikes = sankaku_contents.get("favourite", None)
    boccModel.score.BooruSumLikes = sankaku_contents.get("favourite", None)
    return boccModel


def gv2_resolver(gv2_lines: list[str], boccModel: model.ImageMeta):
    grb_dict = {}
    for lin in gv2_lines:
        mapping = lin.split(": ")[0]
        content = ": ".join(lin.split(": ")[1:])
        grb_dict[mapping] = content

    boccModel.tags.Booru = grb_dict["GE"].split(" ")
    boccModel.chars.Booru = grb_dict["CH"].split(" ")
    boccModel.rating = grb_dict["RT"].split(" ")[0]
    if not boccModel.extra:
        boccModel.extra = model.Extra()
    boccModel.extra.uploader = grb_dict["AU"]
    boccModel.extra.artists = grb_dict["AT"].split(" ")
    return boccModel


def chunker(n, iterable):
    """chunker(3, 'ABCDEFG') --> ('A', 'B', 'C'), ('D', 'E', 'F'),  ('G',)"""
    fillvalue = (
        object()
    )  # Anonymous sentinel object that can't possibly appear in input
    args = (iter(iterable),) * n
    for x in zip_longest(*args, fillvalue=fillvalue):
        if x[-1] is fillvalue:
            # takewhile optimizes a bit for when n is large and the final
            # group is small; at the cost of a little performance, you can
            # avoid the takewhile import and simplify to:
            # yield tuple(v for v in x if v is not fillvalue)
            yield tuple(takewhile(lambda v: v is not fillvalue, x))
        else:
            yield x


try:
    import orjson as json

    orig_dump = json.dumps

    def orjson_dumps(_obj, **kwargs) -> bytes:
        return orig_dump(_obj, option=json.OPT_INDENT_2)

    json.dumps = orjson_dumps

except ImportError:
    print(
        "[KessokuTaggers] orjson not installed. Consider installing for improved deserialization performance."
    )
    import json

    orig_dump = json.dumps

    def json_dumps(**kwargs) -> bytes:
        return orig_dump(**kwargs, ensure_ascii=False).encode("utf-8")

    json.dumps = json_dumps

json_dump_fn = orjson_dumps
json_load_fn = json.loads

IMAGE_SUFFIXES = set([".jpg", ".jpeg", ".png", ".webp"])


def folder_images(folder: pathlib.Path, recurse=True):
    for suffix in IMAGE_SUFFIXES:
        if recurse:
            for file in folder.rglob(f"*{suffix}"):
                yield file
        else:
            for file in folder.glob(f"*{suffix}"):
                yield file


def get_image_files(path: pathlib.Path, recurse=False):
    if path.is_dir():
        files = folder_images(path, recurse=recurse)
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            files = [path]
        else:
            raise Exception(f"path: {path} suffix is not {IMAGE_SUFFIXES}.")
    if not path.is_dir() and not path.is_file():
        raise Exception(f"path: {path} is not a file or directory")
    return files

# Convers various booru formats into bocchi compatible files.
import json

import pathlib
import typer
import tqdm
from library import utils
from library import model as BocchiModel

app = typer.Typer()

@app.command()
def grabber(path: pathlib.Path, replace: bool = False, recurse: bool = False):
    # does .gv2.txt file format which looks like this:
    # 
    # TA: edo_mond_(edoedoedomond) guel_jeturk miorine_rembran shaddiq_zenelli gundam gundam_suisei_no_majo highres tagme 1girl 2boys multiple_boys
    # AU: bubcus93
    # AT: edo_mond_(edoedoedomond)
    # CO: gundam
    # CH: guel_jeturk+miorine_rembran+shaddiq_zenelli
    # GE: 1girl, 2boys, multiple_boys
    # SC: 4
    # RT: general

    # Resolve filepaths
    files = utils.get_image_files(path, recurse=recurse)

    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:
        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if not replace and meta_file.exists():
            raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(raw_meta.get("rating"), list):
                raw_meta["rating"] = raw_meta["rating"][0]
            meta = BocchiModel.ImageMeta.from_dict(raw_meta)
            # print(meta)
        else:
            tag = BocchiModel.Tags()
            char = BocchiModel.Chars()
            meta = BocchiModel.ImageMeta(tags=tag, chars=char)
        grabber_v2 = file.with_suffix(file.suffix + ".gv2.txt")
        if grabber_v2.exists():
            contents = grabber_v2.read_text(encoding="utf-8").split("\n")
            meta = utils.gv2_resolver(contents, meta)

        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

@app.command()
def sankaku(path: pathlib.Path, replace: bool = False, recurse: bool = False):
    # Command is meant for Drumroll.py which is a personal scraper.

    # Resolve filepaths
    files = utils.get_image_files(path, recurse=recurse)

    if not isinstance(files, list):  # Either generator or list
        files = tqdm.tqdm(files, unit="files")
    for file in files:

        meta_file = file.with_suffix(file.suffix.lower() + ".boc.json")
        if not replace and meta_file.exists():
            raw_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(raw_meta.get("rating"), list):
                raw_meta["rating"] = raw_meta["rating"][0]
            meta = BocchiModel.ImageMeta.from_dict(raw_meta)
            # print(meta)
        else:
            tag = BocchiModel.Tags()
            char = BocchiModel.Chars()
            meta = BocchiModel.ImageMeta(tags=tag, chars=char)

        json_file = file.with_suffix(".json")
        if json_file.exists():
            json_contents = json.loads(json_file.read_text(encoding="utf-8"))
            if "tags" in json_contents and "file_url" in json_contents:
                # Check what site
                if "sankaku" in json_contents["file_url"]:
                    meta = utils.drumroll_resolver(json_contents, meta)
                else:
                    raise Exception(f"{file} does not have a compatible .json file")
        else:
            raise Exception(f"{file} does not have either a gv2 file or a compatible .json file.")

        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        

if __name__ == "__main__":
    app()

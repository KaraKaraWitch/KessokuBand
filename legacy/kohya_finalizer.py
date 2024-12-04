# Format pleaser.
import json
import pathlib
import random
from PIL import Image
import click

junk = [
    "anime_coloring",
    "parody",
    "virtual_youtuber",
    "official_style",
]

@click.command("finalize")
@click.argument("folder")
@click.option("--skip_charas", "-s", is_flag=True)
@click.option("--set_trigger", "-t", default="")
def finalize(folder, skip_charas: bool, set_trigger: str):
    if set_trigger:
        print(f"Using trigger word: \"{set_trigger}\"")
    if skip_charas:
        print("Skipping adding character names...")
    train_dir = pathlib.Path(folder)
    for folder in train_dir.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.suffix.endswith("jpeg") or file.suffix.endswith("jpg") or file.suffix.endswith("png"):
                    if file.suffix.endswith(".png"):
                        chk = False
                        with Image.open(file) as im:
                            if chk:
                                print(file,im.mode)
                            if im.mode == "RGBA" or im.info.get("transparency"):
                                print("Flatten:", file)
                                if im.format != "RGBA":
                                    # Transparent Paletted
                                    im = im.convert("RGBA")
                                
                                new_image = Image.new("RGBA", im.size, "WHITE")

                                new_image.paste(im, mask=im)
                                im.close()
                                new_image = new_image.convert("RGB")
                                new_image.save(file,"PNG")
                    json_data = file.with_suffix(f"{file.suffix}.json")
                    caption = file.with_suffix(".txt")
                    if json_data.exists():
                        cont = json.loads(json_data.read_text(encoding="utf-8"))
                        tags_list = set()
                        if isinstance(cont, dict):
                            tags_list = set()
                            for k,v in cont.items():
                                if "tags" in k and isinstance(v, list):
                                    tags_list.update(set(v))
                            popd = []
                            for tag in tags_list:
                                if tag.startswith("rating:"):
                                    popd.append(tag)
                            for pop in popd:
                                print("pop")
                                tags_list.remove(pop)
                            tags_list = list(tags_list)
                            random.shuffle(tags_list)

                            if not skip_charas:
                                if "charas" in cont:
                                    tags_list = cont["charas"] + tags_list
                            for tag in tags_list.copy():
                                if tag in junk:
                                    tags_list.remove(tag)
                            tags_list = [tag.replace("_"," ") for tag in tags_list]
                            if set_trigger:
                                tags_list = [set_trigger] + tags_list
                            
                        else:
                            raise NotImplementedError
                        caption.write_text(", ".join(tags_list))

if __name__ == "__main__":
    finalize()
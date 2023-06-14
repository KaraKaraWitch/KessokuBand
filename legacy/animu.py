import pathlib
import sys

for text_file in pathlib.Path(sys.argv[1]).resolve().rglob("*.txt"):
    tags = text_file.read_text(encoding="utf-8").split(",")
    tags = [tag.strip() for tag in tags] + ["anime"]
    text_file.write_text(", ".join(tags),encoding="utf-8")
    print(text_file.name)

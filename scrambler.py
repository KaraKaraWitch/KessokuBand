import pathlib, uuid, sys

def main(path: pathlib.Path):
    
    for file in path.rglob("*.jpg"):
        if file.is_file():
            file.rename(file.with_stem(str(uuid.uuid4())))
    for file in path.rglob("*.png"):
        if file.is_file():
            file.rename(file.with_stem(str(uuid.uuid4())))
    pass


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(pathlib.Path(arg).resolve())
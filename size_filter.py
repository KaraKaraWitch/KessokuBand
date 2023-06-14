
import PIL.Image, tqdm, PIL
import pathlib, sys



min_sz = 512 * 512
def main(path: pathlib.Path):

    with tqdm.tqdm(desc=f"Filtering: ?", dynamic_ncols=True,unit="file") as pbar:
        for file in path.iterdir():
            try:
                with PIL.Image.open(file) as im:
                    pxsz = im.size[0] * im.size[0]
                    if pxsz > 6000 * 6000:
                        raise Exception("Size too infinite?")
                    try:
                        im.resize((1,1),resample=PIL.Image.NEAREST)
                    except Exception as e:
                        print(f"{e} Resize Error")
                        file.unlink()
                
            except PIL.UnidentifiedImageError:
                file.unlink()
                continue
            except Exception as e:
                print(f"{e}")
                file.unlink()
            if pxsz <= min_sz:
                #print(file, "too small")
                file.unlink()
            pbar.update(1)

if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve())
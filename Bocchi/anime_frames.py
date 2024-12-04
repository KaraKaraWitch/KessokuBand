import enum
import json
import pathlib
import secrets
import shutil
import subprocess
import tarfile
import typing

import httpx
import natsort
import orjson
import typer

app = typer.Typer()
scxvid_path = pathlib.Path(__file__).resolve().parent.parent / "Tools" / "scxvid.exe"


def process_kframes(source_file: pathlib.Path, output_folder: pathlib.Path, fps=10):
    fs = kframe_file = output_folder / "kframes.log"
    if fs.is_file():
        return kframe_file
    processor = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            str(source_file.resolve()),
            "-f",
            "yuv4mpegpipe",
            "-vf",
            f"select=not(mod(n\\,{fps}))",
            "-pix_fmt",
            "yuv420p",
            "-vsync",
            "drop",
            "-",
        ],
        stdout=subprocess.PIPE,  # ,stderr=subprocess.STDOUT
    )

    scx = subprocess.Popen(
        [str(scxvid_path), kframe_file],
        stdin=processor.stdout,
    )
    # Wait for both processes to be done
    processor.wait()
    scx.wait()
    if kframe_file.exists() and len(kframe_file.read_bytes()) > 0:
        return kframe_file
    return False


def get_subtitle(source_file, output_folder, fps=10):
    json_data = subprocess.check_output(
        [
            "ffprobe",
            "-i",
            str(source_file.resolve()),
            "-print_format",
            "json",
            "-show_streams",
            "-v",
            "quiet",
        ]
    )
    subs = []
    for stream in json.loads(json_data)["streams"]:
        if stream["codec_type"] == "subtitle":
            if subs["disposition"]["default"]:
                return


def extract_frames(source_file, output_folder, fps=10, format="qoi"):
    if format == "jpg":
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(source_file.resolve()),
                "-vf",
                f"select=not(mod(n\\,{fps}))",
                "-qscale:v",
                "1",
                "-qmin",
                "1",
                "-qmax",
                "1",
                "-vsync",
                "vfr",
                f"{output_folder}/frame_%04d.jpg",
            ]
        )
    elif format == "qoi":
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(source_file.resolve()),
                "-vf",
                f"select=not(mod(n\\,{fps}))",
                f"{output_folder}/frame_%04d.qoi",
            ]
        )

def get_selected_frames(kframe_file: pathlib.Path, fps: int, mode=0):
    meta = {"mode": mode, "frames": []}
    file_frames = kframe_file.read_text().split("\n")[3:]
    if mode not in range(0,3):
        raise Exception("Expected mode to be 0, 1 or 2.")
    print("mode:", mode, "selected.")

    magics = {
        # Scales the average by that amount (aka offset)
        "frame_step": fps,
    }
    if mode == 0:
        # KeyFrames only
        marks = set()
        for frame_index, fdata in enumerate(file_frames):
            # print(fdata)
            if not fdata:
                continue
            curr_index = frame_index + 1
            if fdata[0] == "i":
                marks.add(curr_index)
                if curr_index not in marks:
                    sub_seconds = curr_index / fps
                    meta["frames"].append(
                        [
                            "key",
                            curr_index,
                            round(sub_seconds, 4),
                            ["Key Frame (Mode 0)"],
                        ]
                    )
    elif mode == 1:
        # KeyFrames + Previous Frame
        marks = set()
        for frame_index, fdata in enumerate(file_frames):
            # print(fdata)
            if not fdata:
                continue
            curr_index = frame_index + 1
            if fdata[0] == "i":
                if curr_index > 1:
                    # Add the frame just before, for animation this represents the final frames typically.
                    # Inbetween frames can some times be kinda derpy. So we reject those.
                    prev_frame = curr_index - 1

                    if prev_frame not in marks:
                        marks.add(prev_frame)
                        sub_seconds = prev_frame / fps
                        print(meta["frames"])
                        meta["frames"].append(
                            [
                                "p_key",
                                prev_frame,
                                round(sub_seconds, 4),
                                ["Previous Key Frame (Mode 1)"],
                            ]
                        )
                if curr_index not in marks:
                    marks.add(curr_index)
                    sub_seconds = curr_index / fps
                    print(meta["frames"])
                    meta["frames"].append(
                        [
                            "key",
                            curr_index,
                            round(sub_seconds, 4),
                            ["Key Frame (Mode 1)"],
                        ]
                    )
    elif mode == 2:
        # KeyFrames + Previous Frame + kblks thresholding
        magics = {
            # Pull in
            **magics,
            # Scales the average by that amount (aka offset)
            "avg_k_scaling": 1.1,
            # The min blk to be accepted for low / close to zero thesholds.
            "min_kblk": 60,
            # No of min frames to consider a "Scene"
            "min_scene_frames": 3,
            # Skip maxed out P_frames from average calculations
            "sk_hilt": False,
        }
        marks = set()
        last_index = -1
        for frame_index, fdata in enumerate([""] + file_frames):
            # print(fdata)
            if not fdata:
                continue
            print(frame_index, fdata)
            if fdata[0] == "i":
                if last_index != -1 and (last_index + 1) + magics[
                    "min_scene_frames"
                ] <= (frame_index - 1):
                    # print("intracheck", last_index, frame_index - 1)
                    # Extract a list of frames
                    kblks = 0

                    intras = file_frames[last_index : frame_index - 1]
                    if len(intras) <= 0:
                        raise Exception("Empty list of Intras?")
                    acc_kblks = 0
                    for p_frame in intras:
                        f_type, _, kblk, _, _, _, _ = p_frame.split(" ")
                        if int(kblk) == 8160:
                            if f_type == "i":
                                raise Exception()
                            elif f_type != "p" and magics["sk_hilt"]:
                                print("Skipping Unusual P_Frame:", p_frame)
                                continue
                        kblks += int(kblk)
                        acc_kblks += 1
                    avg_k = kblks / acc_kblks
                    print(intras, avg_k)
                    for sidx, p_frame in enumerate(intras):
                        _, _, kblk, _, _, _, _ = p_frame.split(" ")
                        kblk = int(kblk)
                        # Use KScaling factor for avg kblks that exceed min_kblk
                        is_kscale = (kblk > (avg_k * magics["avg_k_scaling"])) and max(
                            avg_k, magics["min_kblk"]
                        ) > magics["min_kblk"]
                        # Use KScaling factor for avg kblks that exceed min_kblk
                        is_minb = (
                            kblk > magics["min_kblk"]
                            and max(avg_k, magics["min_kblk"]) <= magics["min_kblk"]
                        )
                        if is_kscale or is_minb:
                            sub_inter_idx = sidx + last_index + 1
                            if sub_inter_idx not in marks:
                                marks.add(sub_inter_idx)
                                sub_seconds = sub_inter_idx / fps
                                print("Add Inter Frame (Mode 2)", sub_inter_idx)
                                meta["frames"].append(
                                    [
                                        "inter",
                                        sub_inter_idx,
                                        round(sub_seconds, 4),
                                        [
                                            f"Inter Frame (Mode 2) KScale: {is_kscale}, MinBlk: {is_minb}",
                                            kblk,
                                            avg_k * magics["avg_k_scaling"],
                                        ],
                                    ]
                                )
                if frame_index not in marks:
                    marks.add(frame_index)
                    sub_seconds = frame_index / fps
                    meta["frames"].append(
                        [
                            "key",
                            frame_index,
                            round(sub_seconds, 4),
                            ["Key Frame (Mode 2)"],
                        ]
                    )
                    print("Add Key Frame (Mode 2)", frame_index)
                    last_index = frame_index
                if frame_index > 1 and (frame_index - 1) not in marks:
                    prev_idx = frame_index - 1
                    marks.add(prev_idx)
                    sub_seconds = prev_idx / fps
                    print("Add P Key Frame (Mode 2)", prev_idx)
                    meta["frames"].append(
                        [
                            "p_key",
                            prev_idx,
                            round(sub_seconds, 4),
                            ["P Key Frame (Mode 2)"],
                        ]
                    )
                    # last_index = prev_idx
                print("Frame work (Mode 2)", frame_index, marks)
    meta["magics"] = magics
    # print(meta)
    return meta


ani_query = {
    "query": "query($page:Int = 1 $id:Int $type:MediaType $isAdult:Boolean = false $search:String $format:[MediaFormat]$status:MediaStatus $countryOfOrigin:CountryCode $source:MediaSource $season:MediaSeason $seasonYear:Int $year:String $onList:Boolean $yearLesser:FuzzyDateInt $yearGreater:FuzzyDateInt $episodeLesser:Int $episodeGreater:Int $durationLesser:Int $durationGreater:Int $chapterLesser:Int $chapterGreater:Int $volumeLesser:Int $volumeGreater:Int $licensedBy:[Int]$isLicensed:Boolean $genres:[String]$excludedGenres:[String]$tags:[String]$excludedTags:[String]$minimumTagRank:Int $sort:[MediaSort]=[POPULARITY_DESC,SCORE_DESC]){Page(page:$page,perPage:20){pageInfo{total perPage currentPage lastPage hasNextPage}media(id:$id type:$type season:$season format_in:$format status:$status countryOfOrigin:$countryOfOrigin source:$source search:$search onList:$onList seasonYear:$seasonYear startDate_like:$year startDate_lesser:$yearLesser startDate_greater:$yearGreater episodes_lesser:$episodeLesser episodes_greater:$episodeGreater duration_lesser:$durationLesser duration_greater:$durationGreater chapters_lesser:$chapterLesser chapters_greater:$chapterGreater volumes_lesser:$volumeLesser volumes_greater:$volumeGreater licensedById_in:$licensedBy isLicensed:$isLicensed genre_in:$genres genre_not_in:$excludedGenres tag_in:$tags tag_not_in:$excludedTags minimumTagRank:$minimumTagRank sort:$sort isAdult:$isAdult){id title{romaji}coverImage{extraLarge large color}bannerImage season seasonYear description type format}}}",
    "variables": {
        "page": 1,
        "type": "ANIME",
        "sort": "SEARCH_MATCH",
        "search": None,
    },
}


class ReleaseType(enum.Enum):
    TV = "TV"
    BD = "BD"


@app.command()
def magic_folder(
    source_folder: pathlib.Path,
    # seq_id:int,
    anime_name: str,
    release_type: ReleaseType,
    output_folder: typing.Optional[pathlib.Path] = None,
    fps: int = 10,
):
    print("Searching anilist for ID...")
    ani_query["variables"]["search"] = anime_name
    r = httpx.post(
        "https://graphql.anilist.co",
        json=ani_query,
        headers={"user-agent": "AnimeFrameTools/1.0.0"},
        follow_redirects=True,
    )
    r.raise_for_status()
    ids = [(z["id"], z["title"]["romaji"]) for z in r.json()["data"]["Page"]["media"]]
    for idx in ids:
        typer.secho(f"{idx[0]}. {idx[1]}")
    anime_idx = typer.prompt("Anime ID", type=int)
    while anime_idx not in [idz[0] for idz in ids]:
        anime_idx = typer.prompt("[Incorrect ID] Anime ID", type=int)

    if not output_folder:
        output_folder = pathlib.Path(__file__).resolve().parent / secrets.token_hex(16)
    for file in source_folder.iterdir():
        if file.suffix.lower() in [".mkv", ".mp4"]:
            process_file(
                file,
                output_folder / file.stem.replace(" ", "_"),
                fps=fps,
                mode=2,
                meta_only=False,
                delete=True,
                tar_wrap=True,
            )
    tar_files = [file for file in output_folder.glob("*.tar")]
    for idx, path in enumerate(natsort.natsorted(tar_files, alg=natsort.ns.PATH)):
        path = pathlib.Path(str(path))
        path.rename(
            path.resolve().parent
            / f"LFAnime-B-{anime_idx}-{str(release_type.value)}-{str(idx+1).zfill(2)}.tar"
        )
    sub_files = [file for file in output_folder.glob("*.ass")]
    for idx, path in enumerate(natsort.natsorted(sub_files, alg=natsort.ns.PATH)):
        path = pathlib.Path(str(path))
        path.rename(
            path.resolve().parent
            / f"LFAnime-B-{anime_idx}-{str(release_type.value)}-{str(idx+1).zfill(2)}.SUB.ass"
        )


@app.command()
def process_folder(
    source_folder: pathlib.Path,
    output_folder: pathlib.Path,
    fps: int = 15,
    mode: int = 1,
    meta_only: bool = False,
    delete: bool = True,
    tar_wrap: bool = False,
):
    for file in source_folder.iterdir():
        if file.suffix.lower() in [".mkv", ".mp4"]:
            process_file(
                file,
                output_folder / file.stem.replace(" ", "_"),
                fps=fps,
                mode=mode,
                meta_only=meta_only,
                delete=delete,
                tar_wrap=tar_wrap,
            )


@app.command()
def process_file(
    source_file: pathlib.Path,
    output_folder: pathlib.Path,
    fps: int = 15,
    mode: int = 1,
    meta_only: bool = False,
    delete: bool = True,
    tar_wrap: bool = False,
):
    if not output_folder.is_dir():
        output_folder.mkdir(exist_ok=True, parents=True)
    if mode not in [0, 1, 2]:
        raise Exception(f"Invalid mode: {mode}")
    kfile = process_kframes(source_file, output_folder, fps=fps)
    if not kfile:
        return
    meta_frames = get_selected_frames(kfile, fps, mode)

    export_fmt = "jpg"
    
    if output_folder.is_dir():
        for file in output_folder.iterdir():
            if file.is_file() and export_fmt in file.suffix.lower():
                file.unlink()
    (output_folder / "metadata.json").write_bytes(
        orjson.dumps(meta_frames, option=orjson.OPT_INDENT_2)
    )
    if meta_only:
        return
    
    
    extract_frames(source_file, output_folder, fps=fps, format=export_fmt)
    print("Generate Restructed Index")
    restructured_index = {}
    for frame_data in meta_frames["frames"]:
        f_type, index, float_time, _ = frame_data
        restructured_index[index] = [f_type, float_time]
    print("Getting Frames")
    if not delete:
        for file in output_folder.glob(f"frame_*.{export_fmt}"):
            frame_file_index = int(file.stem.split("_")[-1].split(".")[0])
            if frame_file_index in restructured_index:
                new_fn = f"{file.stem}_{restructured_index[frame_file_index][0]}_{restructured_index[frame_file_index][1]}{file.suffix}"
                file.rename(file.resolve().parent / new_fn)
            else:
                new_fn = f"{file.stem}_{'DROPPED'}{file.suffix}"
                file.rename(file.resolve().parent / new_fn)
    else:
        t_files = 0
        d_files = 0
        for file in output_folder.glob(f"frame_*.{export_fmt}"):
            frame_file_index = int(file.stem.split("_")[-1].split(".")[0])
            t_files += 1
            if frame_file_index in restructured_index:
                new_fn = f"{file.stem}_{restructured_index[frame_file_index][0]}_{restructured_index[frame_file_index][1]}{file.suffix}"
                file.rename(file.resolve().parent / new_fn)
            else:
                d_files += 1
                file.unlink()
        print(
            f"[Final Stats] {round((d_files / t_files)*100, 4)}% files dropped using mode {mode}. ({d_files} / {t_files})"
        )
    if tar_wrap:
        tar_file = output_folder.resolve().parent / source_file.with_suffix(".tar").name
        with tarfile.TarFile(tar_file, "w") as tar_f:
            for file in output_folder.iterdir():
                with open(file, "rb") as fp:
                    tf = tarfile.TarInfo(file.name)
                    tf.size = file.stat().st_size
                    tar_f.addfile(tf, fileobj=fp)
        shutil.rmtree(output_folder.resolve())
    return output_folder


if __name__ == "__main__":
    app()

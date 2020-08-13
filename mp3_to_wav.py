from typing import Set
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore")


data_folder = Path("data/mp3/train_audio")
wav_foler = Path("data/wav")


def convert_file(file: Path):
    parts = file.parts
    file_name = parts[-1].split(".")[0]
    new_file = wav_foler.joinpath(parts[-2])  # type: Path
    new_file.mkdir(parents=True, exist_ok=True)
    new_file = new_file.joinpath(f"{file_name}.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-n",
            "-i",
            str(file),
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(new_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def ls_dir(input_folder: Path) -> Set[Path]:
    files = []
    for folder in input_folder.iterdir():
        for file in folder.iterdir():
            files.append(file)
    return set(files)


if __name__ == "__main__":
    files = ls_dir(data_folder)
    existing_files = ls_dir(wav_foler)
    files = files - existing_files

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(convert_file, files, chunksize=100)


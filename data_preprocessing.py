from typing import List, Tuple, Set
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import warnings

warnings.filterwarnings("ignore")

import math
import matplotlib.pyplot as plt
import librosa

sample_length = 5  # sample length in seconds

data_folder = Path("data/mp3/train_audio")
spec_folder = Path("data/spectrogram")


def process_file(file_name: Path) -> Tuple[List[List[float]], int]:
    data, FS = librosa.load(file_name)
    result = []

    # number of samples that can be produces from this file
    samples = math.ceil((len(data) / FS) / sample_length)

    window_length = FS * sample_length

    if len(data) < window_length:
        return result, FS

    for i in range(samples):
        # Divide the audio file into sample of length
        # sampel_length. If there are data left at the
        # end the last window will overlap with the
        # previous one.
        start = i * window_length
        end = (i + 1) * window_length
        if end < data.size:
            sample = data[start:end]
        else:
            sample = data[data.size - window_length : data.size]

        result.append(sample)

        # plt.specgram(sample, Fs=FS, NFFT=512, noverlap=0)
        # plt.show()

    return result, FS


def create_spectrograms(file_name: Path):
    data, FS = process_file(file_name)

    bird_folder = get_storage_folder(file_name)
    bird_folder.mkdir(parents=True, exist_ok=True)

    for i, x in enumerate(data):
        plt.specgram(x, Fs=FS, NFFT=512, noverlap=0)
        path = get_storage_file(file_name, bird_folder, i)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis("tight")
        plt.axis("off")
        plt.savefig(path)


def get_storage_folder(path: Path) -> Path:
    parts = path.parts
    folder = parts[-2]
    path = spec_folder.joinpath(folder)
    return path


def get_storage_file(path: Path, prefix: Path, i: int) -> Path:
    parts = path.parts
    name = parts[-1]
    name = name.split(".")[0]
    name = f"{name}_{i}.png"
    path = prefix.joinpath(name)
    return path


def get_existing_files() -> Set[Path]:
    existing_files = []
    for folder in spec_folder.iterdir():
        for file in folder.iterdir():
            parts = file.parts
            folder = parts[-2]
            name = parts[-1]
            name = name.split("_")[0]
            name = f"{name}.mp3"
            path = data_folder.joinpath(folder)
            path = path.joinpath(name)
            existing_files.append(path)

    return set(existing_files)


def data_preprocessing(overwrite: bool = False):
    existing_files = set()
    if not overwrite:
        existing_files = get_existing_files()
    else:
        shutil.rmtree(spec_folder)

    files = []
    for folder in data_folder.iterdir():
        files += list(folder.iterdir())

    files = set(files) - existing_files

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for file in files:
            future = executor.submit(create_spectrograms, file)
            futures.append(future)

        done = False
        num_done = 0
        check_interval = 10
        while not done:
            time.sleep(check_interval)

            done = True
            tmp = 0
            for future in futures:
                done &= future.done()
                if future.done():
                    tmp += 1

            print(f"Done: {tmp}")
            print(
                f"Rate: {float(tmp - num_done) / float(check_interval)} files / second"
            )
            num_done = tmp

            cmd = input()
            if cmd == "q":
                for future in futures:
                    future.cancel()
                break


if __name__ == "__main__":
    data_preprocessing()

from typing import List, Tuple, Set
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
import time
import select
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import math
import librosa
import h5py

sample_length = 10  # sample length in seconds

data_folder = Path("data/mp3/train_audio")
hdf_file_path = Path("data/spectrograms.hdf5")

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128


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

    return result, FS


def create_spectrogram(data: List[float], sr: int) -> np.ndarray:
    return librosa.feature.melspectrogram(
        data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )


def create_spectrograms(file_name: Path) -> Tuple[List[np.ndarray], str, str]:
    data, SR = process_file(file_name)

    specs = []
    for i, x in enumerate(data):
        specs.append(create_spectrogram(x, SR))

    bird_name, mp3_name = _get_bird_name_file(file_name)
    return specs, bird_name, mp3_name


def data_preprocessing(overwrite: bool = False):
    existing_files = set()
    if not overwrite:
        existing_files = _get_existing_files()
    else:
        if hdf_file_path.exists():
            os.remove(hdf_file_path)

    files = []
    for folder in data_folder.iterdir():
        files += list(folder.iterdir())

    files = set(files) - existing_files

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            future = executor.submit(create_spectrograms, file)
            futures.append(future)

        with h5py.File(hdf_file_path, mode="a") as file:
            done = False
            num_done = 0
            check_interval = 5
            last_time = time.time()
            while not done:
                done = True
                tmp = 0
                for future in futures:
                    done &= future.done()
                    if future.done():
                        tmp += 1

                        # get result and safe it into the hdf file
                        specs, bird_name, mp3_name = future.result()
                        grp = file.require_group(bird_name)
                        print(f"{bird_name}/{mp3_name}")
                        grp.create_dataset(mp3_name, data=specs)
                        futures.remove(future)

                num_done += tmp
                print(f"Done: {num_done}, In queue: {len(futures)}")
                print(
                    f"Rate: {float(tmp) / float(check_interval)} files / second"
                )

                if _isInputAvailable():
                    cmd = input()
                    if cmd == "q":
                        for future in futures:
                            future.cancel()
                        break

                s = check_interval - (time.time() - last_time)
                print(f"Sleep time: {s}")
                if s > 0:
                    time.sleep(s)
                last_time = time.time()


def _get_bird_name_file(path: Path) -> Tuple[str, str]:
    parts = path.parts
    return parts[-2], parts[-1]


def _get_existing_files() -> Set[Path]:
    existing_files = []
    with h5py.File(hdf_file_path, mode="r") as file:
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                path = Path(f"{bird_name}/{file_name}")
                path = data_folder.joinpath(path)
                existing_files.append(path)

    return set(existing_files)


def _isInputAvailable():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


if __name__ == "__main__":
    data_preprocessing(overwrite=False)

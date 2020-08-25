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
import librosa
import h5py

sample_length = 5  # sample length in seconds

data_folder = Path("data/mp3/train_audio")
hdf_file_path = Path("data/audio.hdf5")
no_data_file = Path("data/no_data.json")

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128

SR = 22050


def load_file(file_name: Path) -> Tuple[List[float], str, str]:
    try:
        data, sr = librosa.load(str(file_name), sr=SR, mono=True)
    except ZeroDivisionError:
        data = []

    bird_name, mp3_name = _get_bird_name_file(file_name)

    return data, bird_name, mp3_name


def data_preprocessing(overwrite: bool = False):
    existing_files = set()
    if not overwrite:
        existing_files = _get_existing_files()
        num_done = len(existing_files)
    else:
        if hdf_file_path.exists():
            os.remove(hdf_file_path)
        num_done = 0

    files = []
    for folder in data_folder.iterdir():
        files += list(folder.iterdir())

    num_to_progress = len(files)
    files = set(files) - existing_files

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for file in files:
            future = executor.submit(load_file, file)
            futures.append(future)

        with h5py.File(hdf_file_path, mode="a") as file:
            done = False
            check_interval = 5
            last_time = time.time()
            process_error = []
            while not done:
                done = True
                new_done = 0
                for future in futures:
                    done &= future.done()
                    if future.done():
                        new_done += 1

                        # get result and safe it into the hdf file
                        try:
                            print("Fetch result")
                            data, bird_name, mp3_name = future.result(timeout=2)
                            print("Got result")

                            if len(data) > 0:
                                print(f"{bird_name}/{mp3_name}")
                                grp = file.require_group(bird_name)
                                grp.create_dataset(mp3_name, data=data, dtype=np.float32)
                            else:
                                print(f"Process Error: {bird_name}/{mp3_name}")
                                process_error.append(f"{bird_name}/{mp3_name}")
                            print(f"End")
                            futures.remove(future)
                        except TimeoutError:
                            print("Timeout error")

                num_done += new_done
                print(f"Progress: {num_done}/{num_to_progress}")
                print(f"Rate: {float(new_done) / float(check_interval)} files / second")

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
    if hdf_file_path.exists():
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

from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import random
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, wait
import librosa
import librosa.display
import numpy as np

import data_preprocessing

data_folder = Path("data/mp3/train_audio")


def random_choice(max_value: int, chosen: List[int]) -> Tuple[int, List[int]]:
    choice = random.randint(0, max_value)
    while choice in chosen:
        choice = random.randint(0, max_value - 1)
    chosen.append(choice)
    return choice, chosen


def get_bird_name(path: Path) -> str:
    return str(path).split("/")[-2]


def plot_spectrograms(files: List[Path]):
    # use multiple workers to load and preprocess the data
    data = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for file in files:
            future = executor.submit(data_preprocessing.process_file, file)
            futures.append(future)

        wait(futures)

        for future in futures:
            data.append(future.result())

    # plot the spectrograms of the data
    plt.figure(figsize=(18, 14), dpi=200)
    j = 1
    i = 0
    while j < 17 and i != len(data):
        x, sr = data[i]
        i += 1
        if not x:
            continue
        x = random.choice(x)

        plt.subplot(4, 4, j)
        j += 1
        S = data_preprocessing.create_spectrogram(x, sr)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_DB,
            sr=sr,
            hop_length=data_preprocessing.hop_length,
            x_axis="time",
            y_axis="mel",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(get_bird_name(files[i-1]))

    print(f"Spectrogram shape {S.shape}")

    plt.tight_layout()
    plt.show()


def plot_different_birds():
    folder_list = list(data_folder.iterdir())
    folders_list = random.choices(folder_list, k=20)

    files = []
    for folder in folders_list:  # type: Path
        files.append(random.choice(list(folder.iterdir())))

    plot_spectrograms(files)


def plot_same_bird(bird_name: str = None):
    if not bird_name:
        folder_list = list(data_folder.iterdir())
        bird_name = random.choice(folder_list).name

    bird_folder = data_folder.joinpath(bird_name)
    bird_files = list(bird_folder.iterdir())
    files = random.choices(bird_files, k=20)

    plot_spectrograms(files)


if __name__ == "__main__":
    plot_different_birds()

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

from data_preprocessing import process_file

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
            future = executor.submit(process_file, file)
            futures.append(future)

        wait(futures)

        for future in futures:
            data.append(future.result())

    # plot the spectrograms of the data
    plt.figure(figsize=(18, 14), dpi=200)
    for i in range(16):
        x, FS = data[i]
        if not x:
            continue
        x = random.choice(x)

        plt.subplot(4, 4, i+1)
        S = librosa.feature.melspectrogram(x, sr=FS, n_fft=2048, hop_length=512, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=FS, hop_length=512, x_axis="time", y_axis="mel")
        plt.colorbar(format='%+2.0f dB')
        plt.title(get_bird_name(files[i]))

    plt.tight_layout()
    plt.show()


def plot_different_birds():
    folder_list = list(data_folder.iterdir())
    folders_list = random.choices(folder_list, k=20)

    files = []
    for folder in folders_list: # type: Path
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

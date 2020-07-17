from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, wait

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
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(18, 14), dpi=200)

    # use multiple workers to load and preprocess the data
    data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for file in files:
            future = executor.submit(process_file, file)
            futures.append(future)

        wait(futures)

        for future in futures:
            data.append(future.result())

    # plot the spectrograms of the data
    i = 0
    for row in ax:
        for col in row:
            x, FS = data[i]
            x = random.choice(x)
            col.specgram(x, Fs=FS, NFFT=512, noverlap=0)
            col.title.set_text(get_bird_name(files[i]))
            i += 1

    plt.show()


def plot_different_birds():
    folder_list = list(data_folder.iterdir())
    files = random.choices(folder_list, k=16)

    plot_spectrograms(files)


def plot_same_bird(bird_name: str = None):
    if not bird_name:
        folder_list = list(data_folder.iterdir())
        bird_name = random.choice(folder_list).name

    bird_folder = data_folder.joinpath(bird_name)
    bird_files = list(bird_folder.iterdir())
    files = random.choices(bird_files, k=16)

    plot_spectrograms(files)


if __name__ == "__main__":
    plot_same_bird()

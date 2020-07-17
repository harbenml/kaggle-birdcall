from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from random import randint
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, wait

from data_preprocessing import process_file

data_folder = Path("data/mp3/train_audio")


def random_choice(max_value: int, chosen: List[int]) -> Tuple[int, List[int]]:
    choice = randint(0, max_value)
    while choice in chosen:
        choice = randint(0, max_value - 1)
    chosen.append(choice)
    return choice, chosen


def get_files() -> List[Path]:
    folder_list = list(data_folder.iterdir())
    chosen = []
    result = []
    for i in range(16):
        random, chosen = random_choice(len(folder_list), chosen)
        bird = folder_list[random]
        result.append(next(bird.iterdir()))
    return result


def get_bird_name(path: Path) -> str:
    return str(path).split("/")[-2]


def plot_spectrograms():
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(18, 14), dpi=200)

    # use multiple workers to load and preprocess the data
    files = get_files()
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
            col.specgram(x[0], Fs=FS, NFFT=512, noverlap=0)
            col.title.set_text(get_bird_name(files[i]))
            i += 1

    plt.show()


if __name__ == "__main__":
    plot_spectrograms()

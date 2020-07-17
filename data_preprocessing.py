from typing import List, Tuple
from pathlib import Path

import math
import matplotlib.pyplot as plt
import librosa

sample_length = 5  # sample length in seconds


def process_file(file_name: Path) -> Tuple[List[List[float]], int]:
    data, FS = librosa.load(file_name)
    result = []

    # number of samples that can be produces from this file
    samples = math.ceil((len(data) / FS) / sample_length)

    for i in range(samples):
        # Divide the audio file into sample of length
        # sampel_length. If there are data left at the
        # end the last window will overlap with the
        # previous one.
        window_length = FS * sample_length
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


if __name__ == "__main__":
    file = Path("data/mp3/train_audio/sagspa1/XC147491.mp3")
    data, FS = process_file(file)
    pass

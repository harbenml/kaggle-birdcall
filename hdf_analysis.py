from typing import List, Tuple
import h5py
import pandas as pd
from sklearn import preprocessing, utils
import matplotlib.pyplot as plt
import numpy as np

import data_preprocessing


def class_distribution():
    with h5py.File(data_preprocessing.hdf_file_path, "a") as file:
        y = []
        d_type = []
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                y.append(bird_name)

                value = file[f"{bird_name}/{file_name}"]
                if len(value) > 0:
                    d_type.append(value[0].dtype)

                    shape = value[0].shape
                    if shape != (128, 216):
                        print("Wrong shape")
                else:
                    print(f"No data: {value}")
                    del file[f"{bird_name}/{file_name}"]

        print(f"Spectrogram shape {shape}")

        dtype_df = pd.DataFrame(data={"d_type": d_type})
        print(dtype_df["d_type"].unique())
        count(d_type)

        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

        df = pd.DataFrame(data={"class": y})
        print(df.describe())
        df.plot.hist(bins=264)
        plt.show()


def count(d_type: List[str]):
    a = 0
    b = 0
    for i in d_type:
        if i == np.float32:
            a += 1
        elif i == np.float64:
            b += 1
    print(f"float32 count: {a}, float64 count: {b}")


if __name__ == "__main__":
    class_distribution()

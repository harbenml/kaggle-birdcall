from typing import List, Tuple
import h5py
import pandas as pd
from sklearn import preprocessing, utils
import matplotlib.pyplot as plt
import numpy as np

import data_preprocessing


def analyze_hdf():
    with h5py.File(data_preprocessing.hdf_file_path, "a") as file:
        y = []
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                y.append(bird_name)
                value = file[f"{bird_name}/{file_name}"].value
                if len(value) == 0:
                    print("Empty array")

        print(f"Number of files: {len(y)}")


if __name__ == "__main__":
    analyze_hdf()

import h5py
import pandas as pd
from sklearn import preprocessing, utils
import matplotlib.pyplot as plt

import data_preprocessing


def class_distribution():
    with h5py.File(data_preprocessing.hdf_file_path, "r") as file:
        y = []
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                y.append(bird_name)

        label_encoder = preprocessing.LabelEncoder()
        y = label_encoder.fit_transform(y)

        df = pd.DataFrame(data={"class": y})
        print(df.describe())
        df.plot.hist(bins=264)
        plt.show()


if __name__ == "__main__":
    class_distribution()

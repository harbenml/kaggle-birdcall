from typing import Tuple
import h5py
from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split
import torch
import numpy as np

import data_preprocessing


class DataLoader:
    def __init__(self):
        self.file = h5py.File(data_preprocessing.hdf_file_path, mode="r")

        j = 0
        x = []
        y = []
        for bird_name in self.file.keys():
            for file_name in self.file[bird_name].keys():
                path = f"{bird_name}/{file_name}"
                l = self.file[path].value.shape[0]
                for i in range(l):
                    x.append(f"{path}_{i}")
                    y.append(bird_name)
                j += 1

        print(f"{j} files processed")

        self.label_encoder = preprocessing.LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        x, y = utils.shuffle(x, y, random_state=8)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=16
        )

        self.train_state = 0
        self.test_state = 0

    def get_train_batch(self):
        x, y, state = self._get_batch(self.x_train, self.y_train, self.train_state)

        if state < self.train_state:
            state = 0
            self.x_train, self.y_train = utils.shuffle(
                self.x_train, self.y_train, random_state=32
            )

        self.train_state = state
        print(f"Train state: {self.train_state}")
        return x, y

    def get_test_batch(self):
        x, y, state = self._get_batch(self.x_test, self.y_test, self.test_state)

        if state < self.test_state:
            state = 0
            self.x_test, self.y_test = utils.shuffle(
                self.x_test, self.y_test, random_state=64
            )

        self.test_state = state
        print(f"Test state: {self.test_state}")
        return x, y

    def _get_batch(self, x_data, y_data, state):
        x = x_data[state : (state + self.batch_size)]
        y = y_data[state : (state + self.batch_size)]
        state += self.batch_size
        state = state % len(x_data)

        result_x = []
        for path in x:
            path, i = self._split_path(path)
            result_x.append(self.file[path].value[i])

        return result_x, y, state

    def split_path(self, path: str) -> Tuple[str, int]:
        splits = path.split("_")
        return "".join(splits[:-1]), int(splits[-1])

    def close(self):
        self.file.close()


class TrainDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.file = None

    def __len__(self):
        return len(self.dataloader.x_train)

    def __getitem__(self, item):
        with h5py.File(data_preprocessing.hdf_file_path, mode="r") as file:
            path, i = self.dataloader.split_path(self.dataloader.x_train[item])
            return (
                np.repeat(np.expand_dims(file[path].value[i], axis=0), repeats=3, axis=0),
                self.dataloader.y_train[item],
            )


class ValDataLoader(torch.utils.data.Dataset):
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.file = None

    def __len__(self):
        return len(self.dataloader.x_test)

    def __getitem__(self, item):
        with h5py.File(data_preprocessing.hdf_file_path, mode="r") as file:
            path, i = self.dataloader.split_path(self.dataloader.x_test[item])
            return (
                np.repeat(np.expand_dims(file[path].value[i], axis=0), repeats=3, axis=0),
                self.dataloader.y_test[item],
            )


if __name__ == "__main__":
    dl = DataLoader()
    train_dl = TrainDataLoader(dl)
    print(train_dl.__len__())
    print(train_dl.__getitem__(10)[0].shape)

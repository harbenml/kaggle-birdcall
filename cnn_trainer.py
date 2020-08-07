import os
from pathlib import Path

from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import torchvision
import torch
import numpy as np
import pandas as pd

from data_loader import (
    DataLoader,
    TrainDataLoader,
    ValDataLoader,
    CustomImageList,
    train_csv_file,
)


class DataBunchCallback(Callback):
    def __init__(self, data: DataBunch):
        self.data = data

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.data.save("models/data_save.pkl")


# configuration
load_saved_model = False
saved_model_path = Path("models/bestmodel.pth")

# set variables for reproducibility
torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(22)

# train_dl = TrainDataLoader()
# val_dl = ValDataLoader()
# data = ImageDataBunch.create(train_ds=train_dl, valid_ds=val_dl, num_workers=8)
# data.normalize()

df = pd.read_csv(train_csv_file)
src = (
    CustomImageList.from_df(df, ".", cols="x")
    .split_by_rand_pct(valid_pct=0.2)
    .label_from_df(cols="y")
)
data = src.databunch(num_workers=8, bs=64)
data.normalize()

os.environ["TORCH_HOME"] = "models/pretrained"

learn = create_cnn(data, torchvision.models.resnet34, metrics=accuracy, pretrained=True)

if load_saved_model and saved_model_path.exists():
    learn.load(saved_model_path)

learn.loss_func = torch.nn.functional.cross_entropy

callbacks = [SaveModelCallback(learn, monitor="accuracy"), DataBunchCallback(data)]
learn.fit_one_cycle(8, callbacks=callbacks)
learn.unfreeze()
learn.fit_one_cycle(2, callbacks=callbacks)

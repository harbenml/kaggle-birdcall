import librosa
import torchvision
import torchaudio
import fastai
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import h5py


# *****************************************************************************************
sample_length = 5  # time window length in seconds

SR = 22050

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128

data_folder_path = Path(
    "data/mp3/train_audio"
)  # change for training in a Kaggle Notebook
train_csv_path = Path("data/mp3/train.csv")  # change for training in a Kaggle Notebook

hdf_file_path = Path("data/audio.hdf5")  # only for local training

# train config
load_saved_model = False
saved_model_path = Path("models/bestmodel.pth")

kaggle_platform = True  # set true on Kaggle to load data directly from mp3 files
# *****************************************************************************************

# set variables for reproducibility
torch.manual_seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(22)


def preprocess_data() -> pd.DataFrame:
    print("Start preprocessing")
    cache_data_path = Path("data/data.csv")
    if cache_data_path.exists():
        print("Finished preprocessing")
        return pd.read_csv(cache_data_path)

    train_csv = pd.read_csv(train_csv_path)
    train_csv = train_csv.loc[
        (train_csv["ebird_code"] != "lotduc")
        & (train_csv["filename"] != "XC195038.mp3")
    ]

    train_data_dict = {"x": [], "y": []}
    for index, row in train_csv.iterrows():
        duration = row["duration"]
        if duration > sample_length:
            bird = row["ebird_code"]
            file_name = str(row["filename"]).split(".")[0]
            samples = math.ceil(duration / sample_length)
            for i in range(samples):
                train_data_dict["x"].append(f"{bird}/{file_name}.mp3_{i}")
                train_data_dict["y"].append(bird)

    df = pd.DataFrame(data=train_data_dict)
    df.to_csv(cache_data_path)
    print("Finished preprocessing")
    return df


class CustomImageList(ImageList):
    def open(self, fn):
        path, i = self._split_path(fn)

        if kaggle_platform:
            absolute_path = data_folder_path.joinpath(path)
            data, sr = librosa.load(absolute_path, sr=SR, mono=True)
        else:
            with h5py.File(hdf_file_path, "r") as file:
                data = file[path].value

        window_length = SR * sample_length
        start = i * window_length
        end = start + window_length

        # remove leading and trailing zeros from the audio file
        data_trim = np.trim_zeros(data)

        if len(data_trim) < window_length:
            data_final = data[:window_length]
        elif end > len(data_trim):
            data_final = data_trim[len(data_trim) - window_length : len(data_trim)]
        else:
            data_final = data_trim[start:end]

        # mel_spec = librosa.feature.melspectrogram(
        #    data_final, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        # )  # type: np.ndarray

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )(torch.FloatTensor(data_final))
        mel_spec = mel_spec.cpu().detach().numpy()

        # just for debugging reasons
        if mel_spec.shape != (128, 216):
            print(f"Wrong Shape: {path}, Shape: {mel_spec.shape}")
            print(f"data_final length: {len(data_final)}")
            print(f"data length: {len(data)}")

        return np.repeat(np.expand_dims(mel_spec, axis=0), repeats=3, axis=0).astype(
            np.float32
        )

    def _split_path(self, path: str) -> Tuple[str, int]:
        splits = path.split("_")
        path = Path("_".join(splits[:-1]))
        return "/".join(path.parts[-2:]), int(splits[-1])


class DataBunchCallback(Callback):
    def __init__(self, data: DataBunch):
        self.data = data

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.data.save("models/data_save.pkl")


"""
class CustomMelSpec(torch.nn.Module):
    def __init__(self):
        super(CustomMelSpec, self).__init__()

    def forward(self, x):
        return torchaudio.functional.spectrogram(x, pad=0, window)
"""


def train(df: pd.DataFrame) -> Learner:
    print("Start training")
    src = (
        CustomImageList.from_df(df, ".", cols="x")
        .split_by_rand_pct(valid_pct=0.2)
        .label_from_df(cols="y")
    )
    print("Create DataBunch")
    data = src.databunch(num_workers=8, bs=64)
    print("Normalize data")
    data.normalize()

    os.environ["TORCH_HOME"] = "models/pretrained"

    resnet34 = torchvision.models.resnet34(pretrained=True)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # TODO integrate MelSpectrogram calculation into model
    model = torch.nn.Sequential(resnet34)

    learn = Learner(data, model, metrics=accuracy)

    if load_saved_model and saved_model_path.exists():
        learn.load(saved_model_path)

    learn.loss_func = torch.nn.functional.cross_entropy

    callbacks = [SaveModelCallback(learn, monitor="accuracy"), DataBunchCallback(data)]
    learn.fit_one_cycle(8, callbacks=callbacks)
    learn.unfreeze()
    learn.fit_one_cycle(2, callbacks=callbacks)
    print("Finished training")
    return learn


if __name__ == "__main__":
    train_df = preprocess_data()
    learn = train(train_df)
    # TODO write a function to analyze the test data with the trained model

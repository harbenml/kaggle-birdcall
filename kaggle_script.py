import torchvision
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import librosa
import h5py


# *****************************************************************************************
just_test = True  # skip training and use a saved model

sample_length = 5  # time window length in seconds

SR = 22050

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128

# Meta data paths
train_csv_path = Path("data/mp3/train.csv")  # change for training in a Kaggle Notebook
test_csv_path = Path("data/mp3/test.csv")  # change for training in a Kaggle Notebook

# Data paths
hdf_file_path = Path(
    "data/spectrograms_full.hdf5"
)  # change for training in a Kaggle Notebook
test_data_path = Path(
    "data/mp3/example_test_audio"
)  # change for training in a Kaggle Notebook

# train config
load_saved_model = False
saved_model_path = Path("models/bestmodel.pth")
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

    train_data_dict = {"x": [], "y": []}
    with h5py.File(hdf_file_path, "r") as file:
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                shape = file[f"{bird_name}/{file_name}"].shape
                for i in range(shape[0]):
                    train_data_dict["x"].append(f"{bird_name}/{file_name}_{i}")
                    train_data_dict["y"].append(bird_name)

    df = pd.DataFrame(data=train_data_dict)
    df.to_csv(cache_data_path)
    print("Finished preprocessing")
    return df


class CustomImageList(ImageList):
    def open(self, fn):
        path, i = self._split_path(fn)

        with h5py.File(hdf_file_path, "r") as file:
            mel_spec = file[path][i]
            return np.repeat(
                np.expand_dims(mel_spec, axis=0), repeats=3, axis=0
            ).astype(np.float32)

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


def initialize_learner(df: pd.DataFrame) -> Learner:
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

    learn = Learner(data, resnet34, metrics=accuracy)

    if load_saved_model and saved_model_path.exists():
        learn.load(saved_model_path)

    learn.loss_func = torch.nn.functional.cross_entropy
    return learn


def train(df: pd.DataFrame) -> Learner:
    print("Start training")
    learn = initialize_learner(df)

    callbacks = [SaveModelCallback(learn, monitor="accuracy"), DataBunchCallback(data)]
    learn.fit_one_cycle(8, callbacks=callbacks)
    learn.unfreeze()
    learn.fit_one_cycle(2, callbacks=callbacks)
    print("Finished training")
    return learn


def process_file(file_name: Path) -> Tuple[List[List[float]], int]:
    try:
        data, SR = librosa.load(file_name)
    except ZeroDivisionError:
        data = []
        SR = 22050

    # remove leading and trailing zeros from the audio file
    data = np.trim_zeros(data)

    # number of samples that can be produces from this file
    samples = math.ceil((len(data) / SR) / sample_length)

    window_length = SR * sample_length

    result = []
    if len(data) < window_length:
        return result, SR

    for i in range(samples):
        # Divide the audio file into sample of length
        # sampel_length. If there are data left at the
        # end the last window will overlap with the
        # previous one.
        start = i * window_length
        end = (i + 1) * window_length
        if end < data.size:
            sample = data[start:end]
        else:
            sample = data[data.size - window_length : data.size]

        result.append(sample)

    return result, SR


def create_spectrogram(data: List[float], sr: int) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )  # type: np.ndarray
    return mel_spec.astype(np.float32)


def create_spectrograms(file_name: Path) -> List[np.ndarray]:
    data, SR = process_file(file_name)

    specs = []
    for i, x in enumerate(data):
        specs.append(create_spectrogram(x, SR))

    return specs


def prediction_on_test_set(learn: Learner):
    test_df = pd.read_csv(test_csv_path)

    for index, row in test_df.iterrows():
        path = test_data_path.joinpath(f"{row['audio_id']}.mp3")
        specs = create_spectrograms(path)
        # TODO make predictions on spectrograms


if __name__ == "__main__":
    train_df = preprocess_data()
    if not just_test:
        learn = train(train_df)
    else:
        learn = initialize_learner(train_df)

    prediction_on_test_set(learn)

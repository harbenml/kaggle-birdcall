import librosa
import torchvision
from fastai.vision import *
from fastai.callbacks import SaveModelCallback
from pydub import AudioSegment


# *****************************************************************************************
sample_length = 5  # time window length in seconds

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128

data_folder_path = Path("data/wav")
train_csv_path = Path("data/mp3/train.csv")

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

    train_csv = pd.read_csv(train_csv_path)

    train_data_dict = {"x": [], "y": []}
    for index, row in train_csv.iterrows():
        duration = row["duration"]
        if duration >= sample_length:
            bird = row["ebird_code"]
            file_name = str(row["filename"]).split(".")[0]
            samples = math.ceil(duration / sample_length)
            for i in range(samples):
                train_data_dict["x"].append(f"{bird}/{file_name}.wav_{i}")
                train_data_dict["y"].append(bird)

    df = pd.DataFrame(data=train_data_dict)
    df.to_csv(cache_data_path)
    print("Finished preprocessing")
    return df


class CustomImageList(ImageList):
    def open(self, fn):
        path, i = self._split_path(fn)

        data, sr = librosa.load(path, sr=None, mono=False)

        window_length = sr * sample_length
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

        mel_spec = librosa.feature.melspectrogram(
            data_final, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )  # type: np.ndarray

        return np.repeat(np.expand_dims(mel_spec, axis=0), repeats=3, axis=0).astype(
            np.float32
        )

    def _split_path(self, path: str) -> Tuple[str, int]:
        splits = path.split("_")
        return "_".join(splits[:-1]), int(splits[-1])


class DataBunchCallback(Callback):
    def __init__(self, data: DataBunch):
        self.data = data

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.data.save("models/data_save.pkl")


def train(df: pd.DataFrame) -> Learner:
    print("Start training")
    src = (
        CustomImageList.from_df(df, str(data_folder_path), cols="x")
        .split_by_rand_pct(valid_pct=0.2)
        .label_from_df(cols="y")
    )
    print("Create DataBunch")
    data = src.databunch(num_workers=8, bs=64)
    print("Normalize data")
    data.normalize()

    os.environ["TORCH_HOME"] = "models/pretrained"

    learn = cnn_learner(
        data, torchvision.models.resnet34, metrics=accuracy, pretrained=True
    )

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
    train(train_df)

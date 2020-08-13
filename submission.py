from pathlib import Path

from fastai.vision import *
import librosa
import torch
import pandas as pd

from data_preprocessing import create_spectrogram, process_file


def load_spectrogram(file_name: Path, offset: int, duration: int = 5) -> np.ndarray:
    data, sr = librosa.load(str(file_name), offset=offset, duration=duration)
    spec = create_spectrogram(data, sr)
    return np.repeat(np.expand_dims(spec, axis=0), repeats=3, axis=0)


def process_prediction_site_12(
    learn: Learner, pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> str:
    outs = pred[2].numpy()
    indices = (outs >= 0.04).nonzero()
    i2c = {v: k for k, v in learn.data.c2i.items()}
    return " ".join([i2c[i] for i in indices[0]])


def process_prediction_site_3(
    learn: Learner, preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    i2c = {v: k for k, v in learn.data.c2i.items()}
    indices = np.asarray([i[1].numpy() for i in preds], dtype=np.int)
    pred = np.argmax(np.bincount(indices))
    return i2c[pred]


def main(learn: Learner):
    test_csv_file = Path("data/mp3/test.csv")
    test_df = pd.read_csv(test_csv_file)

    submission = {"row_id": [], "birds": []}
    for audio_id, data in test_df.groupby("audio_id"):
        site = data.iloc[0]["site"]
        file_name = Path(
            "data/mp3/example_test_audio/BLKFR-10-CPL_20190611_093000.pt540.mp3"
        )  # Path(f"{audio_id}.mp3")
        if site == "site_1" or site == "site_2":
            for index, row in data.iterrows():
                offset = row["seconds"]
                spec = load_spectrogram(file_name, offset)
                pred = learn.predict(spec)
                # pred_str = process_prediction_site_12(learn, pred)
                submission["row_id"].append(row["row_id"])
                submission["birds"].append(pred[0])
        elif site == "site_3":
            x, sr = process_file(file_name)
            preds = []
            for v in x:
                spec = np.repeat(np.expand_dims(create_spectrogram(v, sr), axis=0), repeats=3, axis=0)
                pred = learn.predict(spec)
                preds.append(pred)
            pred = process_prediction_site_3(learn, preds)
            submission["row_id"].append(data.iloc[0]["row_id"])
            submission["birds"].append(pred)

    sub_df = pd.DataFrame(data=submission)
    print(sub_df)
    sub_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    learn = load_learner("models/")
    main(learn)

from typing import List, Tuple, Set
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
import time
import select
import sys
import warnings
import json
import subprocess

warnings.filterwarnings("ignore")

import numpy as np
import math
import librosa
from pydub import AudioSegment
import h5py

sample_length = 5  # sample length in seconds

data_folder = Path("data/mp3/train_audio")
hdf_file_path = Path("data/spectrograms.hdf5")
no_data_file = Path("data/no_data.json")

# spectrogram config
n_fft = 2048
hop_length = 512
n_mels = 128


def process_file(file_name: Path) -> Tuple[List[List[float]], int]:
    SR = 22050
    sound = AudioSegment.from_mp3(file_name)
    sound = sound.set_frame_rate(SR)
    data0 = sound.get_array_of_samples()
    data = np.asarray(data0, dtype=np.float32)

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


def create_spectrograms(file_name: Path) -> Tuple[List[np.ndarray], str, str]:
    data, SR = process_file(file_name)

    specs = []
    for i, x in enumerate(data):
        specs.append(create_spectrogram(x, SR))

    bird_name, mp3_name = _get_bird_name_file(file_name)
    return specs, bird_name, mp3_name


def data_preprocessing(overwrite: bool = False):
    existing_files = set()
    if not overwrite:
        existing_files, no_data = _get_existing_files()
        num_done = len(existing_files)
    else:
        if hdf_file_path.exists():
            os.remove(hdf_file_path)
        num_done = 0
        no_data = []

    files = []
    for folder in data_folder.iterdir():
        files += list(folder.iterdir())

    num_to_progress = len(files)
    files = set(files) - existing_files

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            future = executor.submit(create_spectrograms, file)
            futures.append(future)

        with h5py.File(hdf_file_path, mode="a") as file:
            done = False
            check_interval = 5
            last_time = time.time()
            while not done:
                done = True
                new_done = 0
                for future in futures:
                    done &= future.done()
                    if future.done():
                        new_done += 1

                        # get result and safe it into the hdf file
                        try:
                            print("Fetch result")
                            specs, bird_name, mp3_name = future.result(timeout=2)
                            print("Got result")

                            # make sure there are actual data
                            if len(specs) > 0:
                                print(f"{bird_name}/{mp3_name}")
                                grp = file.require_group(bird_name)
                                grp.create_dataset(
                                    mp3_name, data=specs, dtype=np.float32
                                )
                                print(f"End")
                            else:
                                print(f"Empyt file: {bird_name}/{mp3_name}")
                                no_data.append(f"{bird_name}/{mp3_name}")
                            futures.remove(future)
                        except TimeoutError:
                            print("Timeout error")

                num_done += new_done
                print(f"Progress: {num_done}/{num_to_progress}")
                print(f"Rate: {float(new_done) / float(check_interval)} files / second")

                if _isInputAvailable():
                    cmd = input()
                    if cmd == "q":
                        for future in futures:
                            future.cancel()
                        break

                s = check_interval - (time.time() - last_time)
                print(f"Sleep time: {s}")
                if s > 0:
                    time.sleep(s)
                last_time = time.time()

        no_data_file.write_text(json.dumps(no_data))


def _get_bird_name_file(path: Path) -> Tuple[str, str]:
    parts = path.parts
    return parts[-2], parts[-1]


def _get_existing_files() -> Tuple[Set[Path], List[str]]:
    existing_files = []
    with h5py.File(hdf_file_path, mode="r") as file:
        for bird_name in file.keys():
            for file_name in file[bird_name].keys():
                path = Path(f"{bird_name}/{file_name}")
                path = data_folder.joinpath(path)
                existing_files.append(path)

    # get empty files
    if no_data_file.exists():
        no_data = json.loads(no_data_file.read_text())
        for file in no_data:
            path = data_folder.joinpath(file)
            existing_files.append(path)
    else:
        no_data = []

    return set(existing_files), no_data


def _isInputAvailable():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


if __name__ == "__main__":
    data_preprocessing(overwrite=False)

"""

Empyt file: lobdow/XC490407.mp3
Progress: 16243/21375
Rate: 1.4 files / second
Sleep time: 4.959249258041382
louwat/XC502062.mp3
whtspa/XC319515.mp3
brwhaw/XC420965.mp3
evegro/XC303573.mp3
rthhum/XC109611.mp3
Progress: 16248/21375
Rate: 1.0 files / second
Sleep time: 4.931268215179443
macwar/XC195794.mp3
brncre/XC209341.mp3
spotow/XC501838.mp3
Empyt file: belspa2/XC120285.mp3
greyel/XC201402.mp3
rthhum/XC371413.mp3



Progress: 16289/21375
Rate: 1.6 files / second
Sleep time: 4.9493794441223145
Empyt file: comgol/XC193097.mp3
olsfly/XC210125.mp3
End
mouchi/XC224196.mp3
End
bushti/XC149391.mp3
End
eastow/XC361029.mp3
End
comred/XC118265.mp3
End
chswar/XC195312.mp3
End
Empyt file: gnwtea/XC232627.mp3
houspa/XC379470.mp3
End
commer/XC485814.mp3
End
yerwar/XC182570.mp3
End
Progress: 16300/21375
Rate: 2.2 files / second
Sleep time: 4.936251640319824
bawwar/XC177836.mp3
End
belspa2/XC311051.mp3
End
Empyt file: sheowl/XC216042.mp3
balori/XC315277.mp3
End
pibgre/XC51135.mp3
End
Empyt file: ruckin/XC344647.mp3
gocspa/XC481151.mp3
End
lesgol/XC327512.mp3
End
bushti/XC179727.mp3
End



killde/XC210719.mp3
End
Fetch result
Got result
norpar/XC457222.mp3
End
Fetch result
Got result
Empyt file: chiswi/XC52369.mp3
Fetch result
Got result
lesyel/XC218182.mp3
End
Fetch result
Got result
Empyt file: fiespa/XC306928.mp3
Fetch result
Got result
cliswa/XC295948.mp3
End
Fetch result
Got result
canwar/XC193417.mp3
End
Fetch result
Got result
Empyt file: wooduc/XC453756.mp3
Fetch result
Got result
Empyt file: amepip/XC406871.mp3
Fetch result

"""

import os
import json
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display
import pandas as pd


def save_stft(folder_path: str, json_path: str) -> None:
    # dictionary to store data
    data = {
        "abs_stft": [],
        "labels": []
    }

    df = pd.read_csv("./.dataset/Y_train_ofTdMHi.csv")

    # loop through the files in the folder
    for i, (dirpath, _, filenames) in enumerate(os.walk(folder_path)):
        for f in filenames:
            if not f.endswith(".wav"):
                continue
            filename = f.split("-")[0]
            # load the audio file
            audio, sr = sf.read(os.path.join(dirpath, f))
            # compute the stft
            tf_sig = lr.stft(audio, n_fft=2048)
            # real_stft = tf_sig.real.tolist()
            # imag_stft = tf_sig.imag.tolist()
            abs_stft = np.abs(tf_sig)
            # store the stft
            # data["stft"].append(tf_sig.tolist())  # convert np.ndarray to list
            # data["stft_real"].append(real_stft)
            # data["stft_imag"].append(imag_stft)
            
            data["abs_stft"].append(abs_stft.tolist())

            # store the label
            data["labels"].append(df[df["id"]==f]["pos_label"].values[0])
    
    # save the data
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("Saved json file")        



if __name__ == "__main__":
    save_stft("./.dataset/X_train_min", "data.json")

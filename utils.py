import json
from tqdm import tqdm
import librosa
import numpy as np
from scipy import signal

def retrieve_hyper_params_from_json(json_file):
    with open(json_file) as f:
        hyper_params = json.load(f)
        N = hyper_params['current_params']['N']
        iss = hyper_params['current_params']['iss']
        lr = hyper_params['current_params']['lr']
        ridge = hyper_params['current_params']['ridge']
        seed = hyper_params['current_params']['seed']
        sr = hyper_params['current_params']['sr']
    return N, iss, lr, ridge, seed, sr


def load_and_preprocess_data(file_paths, target_length): 
    data = []
    for file_path in tqdm(file_paths, desc="Loading and preprocessing data", unit="file"):
        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else: 
            audio = audio[:target_length]

        audio = bandpass_filter(audio, sr)

        data.append(audio)

    print("Done")
    return np.array(data)

def bandpass_filter(audio, sr):
    sos = signal.butter(6, [5000, 100000], 'bandpass', fs=sr, output='sos')
    audio = signal.sosfiltfilt(sos, audio)
    return audio
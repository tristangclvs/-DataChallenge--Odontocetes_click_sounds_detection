import json
from tqdm import tqdm
import librosa
import numpy as np
from scipy import signal
from audiomentations import Compose, PitchShift, TimeStretch, ClippingDistortion, Shift

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


def pitch_shifter(audio_signal, sample_rate): 
    pitch_shift_values = [-2, 2]
    pitch_shift_transforms = [PitchShift(pitch_shift) for pitch_shift in pitch_shift_values]
    augmentations = Compose(pitch_shift_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio

def time_shift(audio_signal, sample_rate):
    shift_values = [0.5]
    shift_transforms = [Shift(min_shift=-shift, max_shift=shift, shift_unit="fraction", rollover=True) for shift in shift_values]
    augmentations = Compose(shift_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio
    
def time_stretcher(audio_signal, sample_rate): 
    time_stretch_values = [0.8, 1.2]
    time_stretch_transforms = [TimeStretch(time_stretch) for time_stretch in time_stretch_values]
    augmentations = Compose(time_stretch_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio

def load_and_preprocess_data_augmented(df, target_length): 
    data_list = []
    labels_list = []

    for index, row in tqdm(df.iterrows(), desc="Loading and preprocessing data", unit="file", total=len(df)):
        audio, sr = librosa.load(row["relative_path"], sr=None)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else: 
            audio = audio[:target_length]

        sos = signal.butter(6, [5000, 100000], 'bandpass', fs=sr, output='sos')
        audio = signal.sosfiltfilt(sos, audio)

        current_audio_label = row["label"]

        original_audio = audio.astype(np.float32)
        data_list.append(original_audio)
        labels_list.append(current_audio_label)

        augmented_audio = None

        augmented_audio_pitch = pitch_shifter(audio, sr).astype(np.float32)
        augmented_audio_timeshift = time_shift(audio, sr).astype(np.float32)


        data_list.append(augmented_audio_pitch)
        data_list.append(augmented_audio_timeshift)
        labels_list.append(current_audio_label)
        labels_list.append(current_audio_label)

    data = np.array(data_list)
    labels = np.array(labels_list)
    
    print("Doneeeeeeeeeeee")
    return data, labels

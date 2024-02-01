import json
from tqdm import tqdm
import librosa
import numpy as np
from scipy import signal
from audiomentations import Compose, PitchShift, TimeStretch, Shift
from tensorflow.keras import layers, models

def retrieve_hyper_params_from_json(json_file):
    """Retrieves the hyper parameters from a json file in case of hyper parameter optimization for reservoir computing."""
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
    """Loads and preprocesses the data from the given file paths, without data augmentation."""
    data = []
    for file_path in tqdm(file_paths, desc="Loading and preprocessing data", unit="file"):
        audio, sr = librosa.load(file_path, sr=None)
        audio = bandpass_filter(audio, sr)
        data.append(audio)
    print("Done")
    return np.array(data)

def bandpass_filter(audio, sr):
    """Applies a bandpass filter to the audio signal."""
    sos = signal.butter(6, [5000, 100000], 'bandpass', fs=sr, output='sos')
    audio = signal.sosfiltfilt(sos, audio)
    return audio


def pitch_shifter(audio_signal, sample_rate):
    """Applies pitch shifting to the audio signal."""
    pitch_shift_values = [-2, 2]
    pitch_shift_transforms = [PitchShift(pitch_shift) for pitch_shift in pitch_shift_values]
    augmentations = Compose(pitch_shift_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio

def time_shift(audio_signal, sample_rate):
    """Applies time shifting to the audio signal."""
    shift_values = [0.5]
    shift_transforms = [Shift(min_shift=-shift, max_shift=shift, shift_unit="fraction", rollover=True) for shift in shift_values]
    augmentations = Compose(shift_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio
    
def time_stretcher(audio_signal, sample_rate): 
    """Applies time stretching to the audio signal."""
    time_stretch_values = [0.8, 1.2]
    time_stretch_transforms = [TimeStretch(time_stretch) for time_stretch in time_stretch_values]
    augmentations = Compose(time_stretch_transforms)
    augmented_audio = augmentations(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio

def load_and_preprocess_data_augmented(df, target_length): 
    """Loads and preprocesses the data from the given file paths contained in dataframe, with data augmentation."""
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

def build_model(target_length):
    print("\nCreating model")
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=9, activation='relu', input_shape=(target_length, 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Second convolutional layer
    model.add(layers.Conv1D(32, kernel_size=9, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Third convolutional layer
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())
    # First fully connected layer
    model.add(layers.Dense(128, activation='relu', kernel_regularizer="l2")) 
    # Second fully connected layer
    model.add(layers.Dense(64, activation='relu', kernel_regularizer="l2")) 
    # Dropout regularization to avoid overfitting
    model.add(layers.Dropout(0.5))
    # Binary classification output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Display the model summary
    model.summary()
    return model
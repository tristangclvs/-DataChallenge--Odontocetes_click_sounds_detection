import pandas as pd
from pathlib import Path
import os
import librosa
import librosa.display
import librosa.feature as feat
import matplotlib.pyplot as plt
import numpy as np
from audiomentations import Compose, PitchShift, TimeStretch, ClippingDistortion
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy import signal
import seaborn as sns
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from time import time

def load_and_preprocess_data(file_paths, target_length): 
    data = []
    for file_path in tqdm(file_paths, desc="Loading and preprocessing data", unit="file"):
        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else: 
            audio = audio[:target_length]

        sos = signal.butter(6, [5000, 100000], 'bandpass', fs=sr, output='sos')
        audio = signal.sosfiltfilt(sos, audio)

        data.append(audio)

    print("Done")
    return np.array(data)

def build_model():
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
    model.add(layers.Dense(128, activation='relu', kernel_regularizer='l2')) #! L2 regularization to remove after
    # Second fully connected layer
    model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2')) #! L2 regularization to remove after
    # Dropout regularization to avoid overfitting
    model.add(layers.Dropout(0.5))
    # Binary classification output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Display the model summary
    model.summary()
    return model

def plot_accuracy():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()


def composer(audio_signal, sample_rate):
    pitch_shift_steps = (-2,2)
    time_stretch_factor = (0.8,1.2)
    clipping_distortion_factor = (0.0, 0.5)

    augment = Compose([
        PitchShift(p=0.3, steps=pitch_shift_steps),
        TimeStretch(p=0.2, factor=time_stretch_factor),
        ClippingDistortion(p=0.3, factor=clipping_distortion_factor)
    ])

    augmented_audio = augment(samples=audio_signal, sample_rate=sample_rate)
    return augmented_audio

# main
if __name__ == "__main__":

    # ====== Set parameters ======
    conv1D_directory = Path.cwd() / "CNN topic" / "Convolution_1D"
    test_directory = Path.cwd() / ".dataset" / "X_test"
    models_directory = Path.cwd() / "CNN topic" / "Convolution_1D" / "models"
    model_name = "1d_cnn_l2.keras"

    # Set the path to the downloaded data
    download_path = Path.cwd() / ".dataset"

    # Audio parameters
    sample_rate = 256000
    audio_duration_seconds = 0.2

    #! ====== Load and preprocess data ====== 
    # Read labels file
    labels_file = download_path / "Y_train_ofTdMHi.csv"
    df = pd.read_csv(labels_file)

    # Construct file path by concatenating folder and file name
    df["relative_path"] = Path(download_path) / "X_train" / df["id"]
    # df["relative_path"] = str(download_path) + "/X_train/" + df["id"]

    # Drop id column (replaced it with relative_path)
    df.drop(columns=["id"], inplace=True)

    df.rename(columns={"pos_label": "label"}, inplace=True)

    # invert relative_path and label columns positions
    df = df[["relative_path", "label"]]
    print(f"### There are {len(df)} audio files in the dataset.")

    table = f"""
    Here is the split into good and bad signals:
    | Label   | Count   |
    |:-------:|:-------:|
    | 0       | {df['label'].value_counts()[0]:.9f} |
    | 1       | {df['label'].value_counts()[1]:.9f} |"""
    print(table, end="\n\n")

    print("Loading and preprocessing data")
    target_length = int(sample_rate * audio_duration_seconds)
    X = load_and_preprocess_data(df["relative_path"], target_length)
    y = df["label"].values.astype(int)

    print("\nSplitting data into train and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_augmented = []

    print("\nAugmenting training data")
    for audio in tqdm(X_train, desc="Augmenting training data", unit="file"):
        X_train_augmented.append(composer(audio, sample_rate))
    X_train_all = np.concatenate((X_train, X_train_augmented))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model = build_model() # Build model

    print("\n------------------ Training model ------------------", end="\n\n")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

    print("\n------------------ Saving model ------------------", end="\n\n")
    os.mkdir(Path(models_directory)) if not os.path.exists(Path(models_directory)) else None
    model.save(Path(models_directory) / model_name)


import pandas as pd
from pathlib import Path
import os
import librosa
import librosa.display
import librosa.feature as feat
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

from AudioUtil import AudioUtil

## Get file paths
# Set the path to the downloaded data
download_path = Path.cwd() / ".dataset"

# Read labels file
labels_file = download_path / "Y_train_ofTdMHi.csv"
df = pd.read_csv(labels_file)

# Construct file path by concatenating folder and file name
df["relative_path"] = str(download_path) + "/X_train/" + df["id"]

# Drop id column (replaced it with relative_path)
df.drop(columns=["id"], inplace=True)

df.rename(columns={"pos_label": "label"}, inplace=True)

# invert relative_path and label columns positions
df = df[["relative_path", "label"]]
print(f"There are {len(df)} audio files in the dataset.")

## _____________________________________________________________________________

audio_util = AudioUtil()

def save_mfccs(nb_files = len(df)):
    audio_util = AudioUtil()
    label_files = np.empty(0)
    audio_mfccs = []
    features_and_labels = []
    print("Starting mfccs generation...")
    for line_num in tqdm(range(nb_files), unit="file", desc="Generating mfccs"):
        file_path = df.loc[line_num, "relative_path"]
        mfccs = audio_util.extract_mfccs(file_path) # , n_mels = 
        audio_mfccs.append(mfccs)
        label_files = np.append(label_files, int(df.loc[line_num, "label"]))
        features_and_labels.append((mfccs, df.loc[line_num, "label"]))
    print("Mfccs generated !", end='\n\n')

    print("Saving mfccs...")
    os.mkdir("numpy_data") if not os.path.exists("numpy_data") else None
    np.save(os.path.join("numpy_data", "audio_mfccs.npy"), audio_mfccs)
    np.save(os.path.join("numpy_data", "label_files.npy"), label_files)
    print("Mfccs saved !")
    features_and_labels = pd.DataFrame(features_and_labels, columns=["mfccs", "label"])

    print("Global shape : ", features_and_labels.shape)
    print(features_and_labels.head())

def get_mfccs_from_file(file_path):
    audio_specs = np.load(file_path)
    return np.array(audio_specs.tolist())

def get_labels_from_file(file_path):
    label_files = np.load(file_path)
    return np.array(label_files.tolist())

def build_model(input_shape):
    # Define the CNN model
    model = tf.keras.Sequential()

    # Add the first convolutional layer
    model.add(tf.keras.layers.Conv2D(64 , kernel_size=(3, 3), activation='relu',
                                    input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    # Flatten the output from the convolutional layers
    model.add(tf.keras.layers.Flatten())

    # Add a fully connected layer for classification
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model




if __name__ == "__main__":
    # save_mfccs()
    print("MFCCs saved !")
    X = get_mfccs_from_file(os.path.join(os.getcwd(),  "numpy_data", "audio_mfccs.npy"))
    y = get_labels_from_file(os.path.join(os.getcwd(), "numpy_data", "label_files.npy"))
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"X : {X[0]}")
    print(f"y : {y[0:10]}")

    # Split data
    X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.75)
    X_train, X_validation, y_train, y_validation=train_test_split(X_train,y_train,train_size=0.8, random_state=64)
    
    print(f"X_train shape : {X_train.shape}")
    # X_train = X_train[..., np.newaxis]  # 4D array -> (num_samples, 130, 13, 1)
    # X_test = X_test[..., np.newaxis]
    # X_validation = X_validation[..., np.newaxis]

    print(X_train.shape)

    # # Build model
    model = build_model(input_shape=(X_train.shape[0], X_train.shape[1], 1))
    model.summary()

    # # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))

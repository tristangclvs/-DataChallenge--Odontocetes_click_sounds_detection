import os
import pandas as pd
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import tensorflow as tf
from codecarbon import EmissionsTracker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')
from utils import load_and_preprocess_data_augmented, build_model


model_name = "data_augmentation_pitch_shift_time_shift_30_epochs.keras"


if __name__ == "__main__":
    tracker = EmissionsTracker(project_name="CNN_topic")
    tracker.start()
    #! ====== Set parameters ======
    conv1D_directory = Path.cwd() / "CNN_topic" / "Convolution_1D"
    test_directory = Path.cwd() / ".dataset" / "X_test"
    models_directory = Path.cwd() /  "../models"
    EPOCHS = 10
    BATCH_SIZE = 32

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

    X, y = load_and_preprocess_data_augmented(df, target_length)

    print(X)
    print("=====================================================")
    print(y)
    os.system('cls')
    print("\nSplitting data into train and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=64)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model = build_model(target_length) # Build model

    print("\n------------------ Training model ------------------", end="\n\n")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[early_stopping])

    print("\n------------------ Saving model ------------------", end="\n\n")
    os.mkdir(Path(models_directory)) if not os.path.exists(Path(models_directory)) else None
    model.save(model_name)
    tracker.stop()

    print("\n------------------ Done, model saved ------------------", end="\n\n")

    print("\n------------------ Plot the confusion matrix ------------------", end="\n\n")
    # plot the confusion matrix
    y_pred = model.predict(X_val)
    y_pred = np.round(y_pred).astype(int)
    y_true = y_val.astype(int)
    cm = tf.math.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.jpg")
    plt.plot()

    print("\n------------------ Done ------------------", end="\n\n")
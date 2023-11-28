import glob
import io
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import IPython.display as ipd
import seaborn as sns
import soundfile as sf
import zipfile
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import efficientnet.tfkeras as efn
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split


# def get_images(samples, sr, output_path):
#     n_fft = 2048
#     hop_length = 512
#     n_mels = 90
#     S = librosa.feature.melspectrogram(y=samples,
#                                        sr=sr,
#                                        n_fft=n_fft,
#                                        hop_length=hop_length,
#                                        n_mels=n_mels,
#                                        fmax=100000)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     fig, ax = plt.subplots(figsize=(2, 2))
#     librosa.display.specshow(S_db,
#                              x_axis='time',
#                              y_axis='linear',
#                              sr=sr,
#                              hop_length=hop_length,
#                              )
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150, format='png',
#                 bbox_inches='tight', pad_inches=0)
#     plt.close()


# input_directory = Path.cwd() / ".dataset/X_test"
# output_directory = "./spectogram_test_images"

# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)


# for filename in tqdm(os.listdir(input_directory)):
#     if filename.endswith(".wav"):  # Check the file extension
#         file_path = os.path.join(input_directory, filename)
#         output_path = os.path.join(
#             output_directory, f"{os.path.splitext(filename)[0]}.png")

#         audio_data, sr = librosa.load(file_path, sr=None)
#         get_images(audio_data, sr, output_path)

# print("Done!")



# Read labels file
labels_file = Path.cwd() / ".dataset" / "Y_random_Xwjr6aB.csv"
df = pd.read_csv(labels_file, encoding='latin1')

# Construct file path by concatenating folder and file name
df["relative_path"] = str(Path.cwd()) + "/X_test/" + df["id"]

# Drop id column (replaced it with relative_path)
df.drop(columns=["id"], inplace=True)

df.rename(columns={"pos_label": "label"}, inplace=True)

# invert relative_path and label columns positions
df = df[["relative_path", "label"]]


# # load data
images_folder = Path.cwd() / 'spectogram_test_images'

X_test = []
for i in tqdm(os.listdir(images_folder)):
    img = image.load_img(os.path.join(images_folder, i),
                         target_size=(255, 255))
    img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)

    X_test.append(img_array)

y_test = df['label'].values

print("Shape of testing dataset: ", len(X_test))

# load model and evaluate
model = tf.keras.models.load_model('trained_model.h5')
model.evaluate(X_test, y_test)
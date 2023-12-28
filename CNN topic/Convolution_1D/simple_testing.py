from tensorflow.keras import models
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
from simple_training import load_and_preprocess_data

def load_test_data(folder_path, target_length):
    file_paths = list(Path(folder_path).rglob('*.wav'))  # Assuming the audio files are in WAV format
    return load_and_preprocess_data(file_paths, target_length)

target_length = int(0.2 * 256000)

conv1D_directory = Path.cwd() / "CNN topic" / "Convolution_1D"
test_directory = Path.cwd() / ".dataset" / "X_test"
models_directory = Path.cwd() / "CNN topic" / "Convolution_1D" / "models"
model_name = "1d_cnn_l2.keras"

X_test = load_test_data(test_directory, target_length)

model = models.load_model(models_directory / model_name)

# predictions = model.predict(X_test)
# binary_predictions = (predictions > 0.5).astype(int)
# print(binary_predictions.shape)
file_names = [file_path.name for file_path in Path(test_directory).rglob('*.wav')]
predictions = model.predict(X_test)

print("----------------------------------")
print(predictions.shape)
print(len(file_names))
print("----------------------------------")

df = pd.DataFrame({'id': file_names, 'pos_label': predictions[:, 0]})
df.to_csv(f"{conv1D_directory}/submission_l2.csv", index=False)
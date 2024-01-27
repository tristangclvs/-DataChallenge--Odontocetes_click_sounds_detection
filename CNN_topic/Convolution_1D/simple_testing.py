from tensorflow.keras import models
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
from simple_training import load_and_preprocess_data
# from training_with_data_augmentation import model_name as model_to_test

def load_test_data(folder_path, target_length):
    file_paths = list(Path(folder_path).rglob('*.wav'))  # Assuming the audio files are in WAV format
    return load_and_preprocess_data(file_paths, target_length)

target_length = int(0.2 * 256000)

conv1D_directory = Path.cwd() / "CNN topic" / "Convolution_1D"
test_directory = Path.cwd() / ".dataset" / "X_test"
models_directory = Path.cwd() / "CNN topic" /  "models"
model_name = "G:\Fac.CAF.AMELI.etc\ENSC\Cours ENSC\Semestre 9\Spé_IA\projet\spe_ia_clics_odontocetes\\1d_cnn_l2_data_augmentation_pitch_shift_time_stretch_30_epochs.h5"

X_test = load_test_data(test_directory, target_length)

model = models.load_model(model_name, compile=False)

file_names = [file_path.name for file_path in Path(test_directory).rglob('*.wav')]
predictions = model.predict(X_test)

print("----------------------------------")
print(predictions.shape)
print(len(file_names))
print("----------------------------------")

df = pd.DataFrame({'id': file_names, 'pos_label': predictions[:, 0]})
df.to_csv(f"{conv1D_directory}/1d_cnn_l2_data_augmentation_pitch_shift_time_stretch_30_epochs.h5", index=False)
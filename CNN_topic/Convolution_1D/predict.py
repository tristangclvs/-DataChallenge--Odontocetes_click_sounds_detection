from tensorflow.keras import models
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
from CNN_topic.Convolution_1D.training import build_model, model_name


##########################
# Audio parameters
sample_rate = 256000
audio_duration_seconds = 0.2
target_length = int(sample_rate * audio_duration_seconds)
model = models.load_model(f"{model_name}")
print("=====================================================")
print(model_name)
print(model.summary())
print("=====================================================", end="\n\n")

##########################

def load_test_data(folder_path, target_length):
    file_paths = list(Path(folder_path).rglob('*.wav'))  # Assuming the audio files are in WAV format
    return load_and_preprocess_data(file_paths, target_length)

target_length = int(0.2 * 256000)

conv1D_directory = Path.cwd() / "CNN topic" / "Convolution_1D"
test_directory = Path.cwd() / ".dataset" / "X_test"
models_directory = Path.cwd() / "CNN topic" /  "models"
model_name = f"G:/Fac.CAF.AMELI.etc/ENSC/Cours ENSC/Semestre 9/Spé_IA/projet/spe_ia_clics_odontocetes/{model_name}"

X_test = load_test_data(test_directory, target_length)

# model = models.load_model(model_name, compile=False)

file_names = [file_path.name for file_path in Path(test_directory).rglob('*.wav')]
predictions = model.predict(X_test)

print("----------------------------------")
print(predictions.shape)
print(len(file_names))
print("----------------------------------")

df = pd.DataFrame({'id': file_names, 'pos_label': predictions[:, 0]})
df.to_csv(f"{model_name.split('.')[0]}.csv", index=False)


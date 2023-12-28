from tensorflow.keras import models
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

def load_and_preprocess_data(file_paths, target_length): 
    data = []
    for file_path in tqdm(file_paths):
        audio, _ = librosa.load(file_path, sr=None)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else : 
            audio = audio[:target_length]
        data.append(audio)
    return np.array(data)

def load_test_data(folder_path, target_length):
    file_paths = list(Path(folder_path).rglob('*.wav'))  # Assuming the audio files are in WAV format
    return load_and_preprocess_data(file_paths, target_length)


target_length = int(0.2 * 256000)

X_test = load_test_data("../../.dataset/X_test", target_length)

model = models.load_model('../models/1d_cnn.h5')

# predictions = model.predict(X_test)
# binary_predictions = (predictions > 0.5).astype(int)
# print(binary_predictions.shape)
file_names = [file_path.name for file_path in Path("../../.dataset/X_test").rglob('*.wav')]
predictions = model.predict(X_test)

df = pd.DataFrame({'id': file_names, 'pos_label': predictions[:, 0]})
df.to_csv('submission.csv', index=False)
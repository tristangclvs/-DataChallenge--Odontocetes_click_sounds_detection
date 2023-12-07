import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

class AudioUtil:
    """Static class for audio processing helper functions."""
    
    @staticmethod
    def open(audio_file: str):
        """Load an audio file. Return the signal as a tensor and the sample rate"""
        sig, sr = librosa.load(audio_file, sr=256000)
        return (sig, sr)
    
    @staticmethod
    def get_audio_duration(sig, sr):
        """Return the duration of an audio signal in seconds"""
        return librosa.get_duration(sig, sr)
    
    @staticmethod
    def mel_spectro_gram(sig: np.array, sr: int, n_mels=32, n_fft=1024):
        """Generate a Spectrogram"""
        # get mel spectrogram
        spec = librosa.feature.melspectrogram(y=sig, sr=sr)
        spec = librosa.amplitude_to_db(spec)
        return spec
    
    @staticmethod
    def extract_mfccs(file_path):
        audio_data, sr = librosa.load(file_path, sr = None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs.T,axis=0)
        return mfccs_scaled_features
    
    @staticmethod
    def get_audio_specs_size(spec):
        """Return the size of a spectrogram image"""
        return spec.shape
    
    @staticmethod
    def plot_mel_spectro_gram(spec: np.array, sr: int):
        """Plot a Spectrogram"""
        # plot mel spectrogram
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(spec, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                                y_axis='mel', sr=sr,
                                ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
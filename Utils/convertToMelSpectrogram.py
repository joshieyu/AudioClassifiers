import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(wav_path, output_path, sample_rate=16000, n_mels=128):
    # Load the audio file
    y, sr = librosa.load(wav_path, sr=sample_rate)
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Convert to log scale (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the mel spectrogram as an image
    plt.figure(figsize=(10, 4))
    plt.imshow(S_dB, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

def convert_wav_to_mel_spectrograms(input_directory, output_directory, sample_rate=16000, n_mels=128):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.wav'):  # Consider only .wav files
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_directory)
                output_file_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.png')
                
                # Convert and save the mel spectrogram
                try:
                    save_mel_spectrogram(input_file_path, output_file_path, sample_rate, n_mels)
                    print(f"Converted {input_file_path} to {output_file_path}")
                except Exception as e:
                    print(f"Failed to convert {input_file_path}: {e}")

# Example usage
input_directory = "./resampledData"
output_directory = "./spectrogramData"

# convert_wav_to_mel_spectrograms(input_directory, output_directory)

save_mel_spectrogram(r'.\eguitarResample.wav', './')
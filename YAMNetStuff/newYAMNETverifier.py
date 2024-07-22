import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import scipy
import scipy.signal

# Load the YAMNet model from TensorFlow Hub
model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the trained classifier model
classifier_model = load_model('yamnet_finetuned_model.h5')

# Re-compile the loaded model to ensure metrics are built
classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define categories
categories = ['acoustic', 'electric', 'lead', 'pad', 'bass']

# Function to read and preprocess audio files
def read_audio(file_path, desired_sample_rate=16000, max_length=16000):
    # Load audio file with librosa
    wav_data, original_sample_rate = librosa.load(file_path, sr=None, mono=True)
    
    # Resample audio to the desired sample rate
    if original_sample_rate != desired_sample_rate:
        desired_length = int(
            round(float(len(wav_data)) / original_sample_rate * desired_sample_rate)
        )
        wav_data = scipy.signal.resample(wav_data, desired_length)
    
    # Pad or truncate audio to ensure uniform length
    if len(wav_data) > max_length:
        wav_data = wav_data[:max_length]
    else:
        wav_data = np.pad(wav_data, (0, max_length - len(wav_data)), 'constant')
    
    return desired_sample_rate, wav_data

# Function to extract features from audio using YAMNet
def extract_features(model, waveform):
    cores, embeddings, spectrogram = model(waveform)
    return embeddings

# Function to predict the category of an audio file and visualize the audio info
def predict_audio_category(file_path):
    sample_rate, wav_data = read_audio(file_path)
    
    # Plot the waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wav_data, sr=sample_rate)
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    
    embedding = extract_features(model_yamnet, wav_data)
    embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension
    predictions = classifier_model.predict(embedding)
    
    predicted_category = categories[np.argmax(predictions)]
    confidence_scores = {categories[i]: predictions[0][i] for i in range(len(categories))}
    
    return predicted_category, confidence_scores

# Example usage
audio_file_path = r'.\resampledData\electric\12-8 Electric Arpeggio 06.wav'
predicted_category, confidence_scores = predict_audio_category(audio_file_path)
print(f'The predicted category for the audio file is: {predicted_category}')
print('Confidence scores for all categories:')
for category, score in confidence_scores.items():
    print(f'{category}: {score:.2f}')

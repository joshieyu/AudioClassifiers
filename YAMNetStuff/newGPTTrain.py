import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import scipy
import scipy.signal

# Example data path
data_path = './resampledData'

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

# Load and preprocess data
audio_data = []
labels = []
categories = ['acoustic', 'electric', 'lead', 'pad', 'bass']

for category in categories:
    category_dir = os.path.join(data_path, category)
    if os.path.isdir(category_dir):  # Check if category_dir is a directory
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            if file_path.endswith('.wav'):  # Check if the file is a .wav file
                # print(file_path)
                _, wav_data = read_audio(file_path)
                audio_data.append(wav_data)
                labels.append(category)

# Convert to DataFrame for further processing
audio_dataframe = pd.DataFrame({'audio_data': audio_data, 'class': labels})

print(audio_dataframe.head())

# Visualize a few audio waveforms
plt.figure(figsize=(14, 10))
for i in range(5):
    plt.subplot(5, 1, i+1)
    librosa.display.waveshow(audio_dataframe['audio_data'][i], sr=16000)
    plt.title(f"Class: {audio_dataframe['class'][i]}")
    plt.tight_layout()
plt.show()

# Convert the audio data to a numpy array
audio_data_np = np.array(audio_dataframe["audio_data"].tolist())

# Convert the class labels to a numpy array
class_labels_np = np.array(audio_dataframe["class"].tolist())

# One-hot encode labels
lb = LabelBinarizer()
labels_one_hot = lb.fit_transform(class_labels_np)


# Visualize the shapes of the numpy arrays
print(f'Shape of audio data numpy array: {audio_data_np.shape}')
print(f'Shape of labels numpy array: {class_labels_np.shape}')
print(f'Shape of one-hot encoded labels numpy array: {labels_one_hot.shape}')

# Load the YAMNet model from TensorFlow Hub
model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# Function to extract features from audio using YAMNet
def extract_features(model, waveform):
    scores, embeddings, spectrogram = model(waveform)
    return embeddings

# Extract features using YAMNet
audio_embeddings = []
for audio in audio_data_np:
    embedding = extract_features(model_yamnet, audio)
    audio_embeddings.append(embedding)

# Convert the embeddings to a numpy array
audio_embeddings_np = np.array(audio_embeddings)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(audio_embeddings_np, labels_one_hot, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the custom model
input_shape = (audio_embeddings_np.shape[1], audio_embeddings_np.shape[2])  # YAMNet outputs 1024-dimensional embeddings

inputs = layers.Input(shape=input_shape)
x = layers.GlobalAveragePooling1D()(inputs)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(lb.classes_), activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('yamnet_finetuned_model.h5')
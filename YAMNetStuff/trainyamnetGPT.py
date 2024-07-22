import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Load the YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet = hub.load(yamnet_model_handle)
yamnet_output = yamnet.signatures['serving_default']

# Function to load and preprocess audio
def load_and_preprocess_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    return waveform

# Function to extract features from audio using YAMNet
def extract_features(yamnet, waveform):
    # Run the model, check the output
    outputs = yamnet_output(tf.constant(waveform, dtype=tf.float32))
    print(outputs.keys())  # Debugging: print the keys of the output dictionary to identify available keys
    embeddings = outputs['embedding']  # Extract the 'embedding' if available
    return embeddings.numpy()

# Path to your dataset
data_dir = './resampledData'

# Load data
labels = []
features = []

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_path.endswith('.wav'):
                waveform = load_and_preprocess_audio(file_path)
                feature = extract_features(yamnet, waveform)
                features.append(feature)
                labels.append(class_name)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the custom model
input_shape = (features.shape[1], features.shape[2])  # YAMNet outputs 1024-dimensional embeddings

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
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('yamnet_finetuned_model.h5')

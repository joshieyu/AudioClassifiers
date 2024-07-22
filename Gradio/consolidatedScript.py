import gradio as gr
from fastai.vision.all import load_learner, PILImage
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
# Load your fastai model

if os.name != 'nt':
    pathlib.WindowsPath = pathlib.PosixPath
    # For Windows
    # learn_inf = load_learner('export.pkl')
# else:
    # learn_inf = load_learner('export.pkl')

# Load your fastai model
learn_inf = load_learner('export.pkl')

# Function to save mel spectrogram and run inference
def save_mel_spectrogram_and_predict(wav_path):
    # Define paths
    output_dir = 'temp_spectrograms'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, 'temp_spectrogram.png')
    
    # Load the audio file
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Save the mel spectrogram as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()
    
    # Run inference on the saved mel spectrogram image
    img = PILImage.create(output_path)
    pred_class, pred_idx, probs = learn_inf.predict(img)
    
    return output_path, {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio interface function
def gradio_interface(audio):
    spectrogram_path, predictions = save_mel_spectrogram_and_predict(audio)
    return spectrogram_path, predictions

# Create the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=[gr.Image(type="filepath", label="Mel Spectrogram"), gr.JSON(label="Class Probabilities")],
    title="Audio Classification with Mel Spectrogram",
    description="Upload an audio file to see its mel spectrogram and classification probabilities."
)

# Launch the interface
interface.launch(share=True)

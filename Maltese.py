from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from datasets import load_dataset
from pydub import AudioSegment
import numpy as np
import librosa

# Load the pre-trained wav2vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h")
model = Wav2Vec2ForCTC.from_pretrained("carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h")

# Make sure to use the device with GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def transcribe_audio(file_path):
    """Function to transcribe audio file to text."""
    
    # Load audio file (use librosa to load the audio)
    audio, rate = librosa.load(file_path, sr=16000)  # wav2vec2 uses a 16kHz sample rate

    # Preprocess the audio for the model
    inputs = processor(audio, return_tensors="pt", padding=True)
    
    # Ensure the model is in eval mode
    model.eval()
    
    # Move inputs to the correct device (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform the transcription
    with torch.no_grad():
        logits = model(input_values=inputs["input_values"]).logits
    
    # Get the predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the predicted ids to text
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

# Example: transcribe an audio file
file_path = "maltese-re.mp3"  # Replace with your file path
transcription = transcribe_audio(file_path)
print("Transcription: ", transcription)

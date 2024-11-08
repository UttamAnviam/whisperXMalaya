from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

app = FastAPI()

# Load the Persian wav2vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-persian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-persian")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read audio file
        audio_data = await file.read()
        audio_path = f"/tmp/{file.filename}"

        # Save the uploaded file temporarily
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # Load the audio with librosa
        audio, rate = librosa.load(audio_path, sr=16000)

        # Preprocess the audio for the model
        inputs = processor(audio, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Transcription
        model.eval()
        with torch.no_grad():
            logits = model(input_values=inputs["input_values"]).logits

        # Get predicted ids and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

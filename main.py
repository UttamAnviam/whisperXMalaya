from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Header
from fastapi.responses import JSONResponse
import whisper
import os
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Determine the device for model inference (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load the Whisper model
model = whisper.load_model("tiny").to(device)

# Constants for chunking
CHUNK_SIZE_MB = 25  # Maximum file size in MB before chunking
CHUNK_LENGTH_MS = 60 * 1000  # 60 seconds per chunk

# Define your expected token
EXPECTED_TOKEN = "your_secure_token_here"

def transcribe_chunk(audio_chunk, chunk_path):
    """ Transcribe a chunk of audio using Whisper and delete the chunk after transcribing. """
    audio_chunk.export(chunk_path, format="wav")
    logging.info(f"Transcribing chunk: {chunk_path}")
    result = model.transcribe(chunk_path)
    os.remove(chunk_path)  # Clean up temporary file
    return result['text']

def process_audio_in_chunks(file_path):
    """ Process large audio files in chunks and return the final transcription. """
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    transcription = ""
    
    with ThreadPoolExecutor() as executor:
        futures = []
        chunk_path_template = "/tmp/temp_chunk_{}.wav"
        
        for i in range(0, total_length_ms, CHUNK_LENGTH_MS):
            chunk_start = i
            chunk_end = min(i + CHUNK_LENGTH_MS, total_length_ms)
            audio_chunk = audio[chunk_start:chunk_end]
            chunk_path = chunk_path_template.format(i)
            futures.append(executor.submit(transcribe_chunk, audio_chunk, chunk_path))

        for future in futures:
            transcription += future.result() + " "

    return transcription

def process_full_audio(file_path):
    """ Transcribe an entire audio file if it's smaller than 25MB. """
    result = model.transcribe(file_path)
    return result['text']

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...), 
    x_token: str = Header(None)  # Custom header for token
):
    EXPECTED_TOKEN = "dummytoken123"
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Save the uploaded audio file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as audio_file:
        content = await file.read()
        audio_file.write(content)

    # Check file size (convert bytes to MB)
    file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
    logging.info(f"Uploaded file size: {file_size_mb:.2f} MB")

    # Process the file and get the transcription
    if file_size_mb > CHUNK_SIZE_MB:
        logging.info("Processing file in chunks...")
        transcription = process_audio_in_chunks(temp_file_path)
    else:
        logging.info("Processing entire file...")
        transcription = process_full_audio(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Return the transcription result
    return JSONResponse(content={"transcription": transcription})

@app.get("/")
def read_root():
    return {"message": "Hello World"}

import os
import hashlib
import logging
import aiofiles
import asyncio
import torch
import whisper
import whisperx
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Determine the device for model inference (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Constants for chunking
CHUNK_SIZE_MB = 25  # Maximum file size in MB before chunking
CHUNK_LENGTH_MS = 60 * 1000  # 60 seconds per chunk

# Define your expected token
EXPECTED_TOKEN = "dummytoken123"

# Load the Whisper model (default)
whisper_model = whisper.load_model("tiny").to(device)

# Load the WhisperX model (for Malayalam)
whisperx_model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type="float32")

# In-memory cache (optional)
cache = {}

def get_file_hash(file_path):
    """ Return the hash of the file to use for caching. """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def save_audio_file(file: UploadFile, temp_file_path: str):
    """ Asynchronously save the uploaded audio file to a temporary file. """
    async with aiofiles.open(temp_file_path, 'wb') as audio_file:
        content = await file.read()
        await audio_file.write(content)

async def transcribe_chunk_async(audio_chunk, chunk_path, model):
    """ Asynchronously transcribe a chunk of audio. """
    audio_chunk.export(chunk_path, format="wav")
    logging.info(f"Transcribing chunk: {chunk_path}")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: model.transcribe(chunk_path))
    os.remove(chunk_path)  # Clean up temporary file
    return result['text']

async def process_audio_in_chunks(file_path, model):
    """ Process large audio files in chunks asynchronously. """
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    transcription = []
    
    chunk_path_template = "/tmp/temp_chunk_{}.wav"
    tasks = []
    
    for i in range(0, total_length_ms, CHUNK_LENGTH_MS):
        chunk_start = i
        chunk_end = min(i + CHUNK_LENGTH_MS, total_length_ms)
        audio_chunk = audio[chunk_start:chunk_end]
        chunk_path = chunk_path_template.format(i)
        tasks.append(transcribe_chunk_async(audio_chunk, chunk_path, model))

    # Gather results asynchronously
    results = await asyncio.gather(*tasks)
    
    # Combine the transcriptions
    transcription = " ".join(results)
    return transcription

def process_full_audio(file_path, model):
    """ Transcribe an entire audio file if it's smaller than 25MB. """
    result = model.transcribe(file_path)
    
    if 'segments' in result:
        transcription = " ".join([segment['text'] for segment in result['segments']])
        return transcription
    else:
        raise ValueError("No transcription segments found.")

@app.post("/filter_transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...), 
    x_token: str = Header(None)  # Custom header for token
):
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Save the uploaded audio file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    await save_audio_file(file, temp_file_path)

    # Check file size (convert bytes to MB)
    file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
    logging.info(f"Uploaded file size: {file_size_mb:.2f} MB")

    # Check cache before processing
    file_hash = get_file_hash(temp_file_path)
    cache_key = f"transcription_{file_hash}"
    cached_transcription = cache.get(cache_key)

    if cached_transcription:
        logging.info("Returning cached transcription.")
        return JSONResponse(content={"transcription": cached_transcription})

    # Detect language using whisperx or whisper
    audio = whisperx.load_audio(temp_file_path)
    language_detection_result = whisperx_model.transcribe(audio, batch_size=4)
    detected_language = language_detection_result['language']

    logging.info(f"Detected language: {detected_language}")

    # Determine model to use based on the detected language
    if detected_language == 'ml':  # 'ml' is the code for Malayalam
        logging.info("Using WhisperX model for Malayalam transcription.")
        model = whisperx_model
    else:
        logging.info("Using default Whisper model for transcription.")
        model = whisper_model

    # Process the file and get the transcription
    if file_size_mb > CHUNK_SIZE_MB:
        logging.info("Processing file in chunks...")
        transcription = await process_audio_in_chunks(temp_file_path, model)
    else:
        logging.info("Processing entire file...")
        transcription = process_full_audio(temp_file_path, model)

    # Cache the result
    cache[cache_key] = transcription  # Fix here: use dictionary assignment

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Return the transcription result
    return JSONResponse(content={"transcription": transcription})

@app.get("/")
def read_root():
    return {"message": "Hello World"}

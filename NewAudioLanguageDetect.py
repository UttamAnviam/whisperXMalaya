from fastapi import FastAPI, HTTPException, Header, Form
from fastapi.responses import JSONResponse
import logging
import os
import torch
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from tempfile import NamedTemporaryFile
import asyncio
import httpx
import whisper

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Constants
CHUNK_SIZE_MB = 25  # Maximum file size in MB before using pipeline
CHUNK_LENGTH_MS = 60 * 1000  # 60 seconds per chunk
EXPECTED_TOKEN = "NgPEbNQnZrKDtRwfaIrBmnryRQZITFhm"

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Initialize both models
whisper_model = None
pipeline_model = None


def load_models():
    global whisper_model, pipeline_model
    logging.info("Loading models...")
    whisper_model = whisper.load_model("medium").to(device)

    pipeline_model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        return_timestamps=True
    )


load_models()

executor = ThreadPoolExecutor()


# Utility Functions
async def split_audio_pipeline(file_path, chunk_duration_ms=180000):
    """Split audio into fixed-size chunks for pipeline processing."""
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        temp_chunk = NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append(temp_chunk.name)
    return chunks


async def transcribe_chunk_pipeline_async(chunk_path):
    """Transcribe a single chunk asynchronously using the pipeline model."""
    try:
        logging.info(f"Transcribing chunk with pipeline: {chunk_path}")
        return pipeline_model(
            chunk_path,
            generate_kwargs={"language": "en"},
            return_timestamps=True
        )
    except Exception as e:
        logging.error(f"Pipeline transcription error: {e}")
        return None


async def process_audio_pipeline_async(file_path):
    """Process large audio files using the pipeline model asynchronously."""
    chunks = split_audio_pipeline(file_path)
    tasks = [transcribe_chunk_pipeline_async(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks)

    transcriptions = []
    for result in chunk_results:
        if result:
            transcriptions.append(result.get('text', ''))

    return " ".join(transcriptions), "en", []  # Segments can be populated if needed


async def transcribe_whisper_sync(file_path):
    """Synchronous transcription using Whisper."""
    result = whisper_model.transcribe(file_path)
    return result['text'], result['language'], result['segments']


async def transcribe_whisper_async(file_path):
    """Asynchronous wrapper for Whisper transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, transcribe_whisper_sync, file_path)



@app.post("/transcribe/")
async def download_and_transcribe(
    url: str = Form(...),
    x_token: str = Header(None)
):
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    filename = url.split("/")[-1]
    temp_file_path = f"/tmp/{filename}"

    try:
        # Download file from the URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download file")

        with open(temp_file_path, "wb") as file:
            file.write(response.content)

        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")

        if file_size_mb > CHUNK_SIZE_MB:
            transcription, language, segments = await process_audio_pipeline_async(temp_file_path)
        else:
            transcription, language, segments = await transcribe_whisper_async(temp_file_path)

        # Construct the `download_url`
        download_url = f"https://process-audio.healthorbit.ai/transcription-results/{filename}"

        return JSONResponse(content={
            "results": [{
                "filename": filename,
                "transcript": {
                    "text": transcription,
                    "segments": segments,
                    "language": language
                },
                "download_url": download_url
            }]
        })

    except Exception as e:
        logging.error(f"Error during download and transcription: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



@app.get("/")
def read_root():
    return {"message": "Whisper Transcription API"}

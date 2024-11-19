from fastapi import FastAPI, UploadFile, File, HTTPException, Header,Form
from fastapi.responses import JSONResponse
import whisper
import os
import torch
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from tempfile import NamedTemporaryFile
import time

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
    
    # Initialize the pipeline with correct parameters
    pipeline_model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        return_timestamps=True  # Enable timestamp generation
    )

# Load models at startup
load_models()

def create_segments_from_chunks(chunks_results, chunk_duration_ms=180000):
    """Create segments from pipeline chunks with timestamps."""
    segments = []
    total_duration = 0
    
    for chunk_idx, chunk_result in enumerate(chunks_results):
        chunk_offset = chunk_idx * chunk_duration_ms / 1000  # Convert to seconds
        
        # If the chunk has chunks (timestamps), process them
        if 'chunks' in chunk_result:
            for chunk in chunk_result['chunks']:
                timestamp = chunk.get('timestamp', None)
                if timestamp:
                    start, end = timestamp
                    # Adjust timestamps based on chunk position
                    start = start + chunk_offset
                    end = end + chunk_offset
                    
                    segment = {
                        'id': len(segments),
                        'start': start,
                        'end': end,
                        'text': chunk['text'],
                        'tokens': [],  # Placeholder for token ids
                        'temperature': 0.0,
                        'avg_logprob': -1.0,
                        'compression_ratio': 1.0,
                        'no_speech_prob': 0.0
                    }
                    segments.append(segment)
                    total_duration = max(total_duration, end)
        else:
            # If no timestamps, create a single segment for the chunk
            segment = {
                'id': len(segments),
                'start': chunk_offset,
                'end': chunk_offset + (chunk_duration_ms / 1000),
                'text': chunk_result['text'],
                'tokens': [],
                'temperature': 0.0,
                'avg_logprob': -1.0,
                'compression_ratio': 1.0,
                'no_speech_prob': 0.0
            }
            segments.append(segment)
            total_duration = max(total_duration, segment['end'])
    
    return segments

def split_audio_pipeline(file_path, chunk_duration_ms=180000):
    """Split audio into fixed-size chunks for pipeline processing."""
    logging.info("Splitting audio into chunks for pipeline...")
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        temp_chunk = NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append(temp_chunk.name)
    return chunks

def transcribe_chunk_pipeline(chunk_path):
    """Transcribe a single chunk using the pipeline model."""
    try:
        logging.info(f"Transcribing chunk with pipeline: {chunk_path}")
        result = pipeline_model(
            chunk_path,
            generate_kwargs={"language": "en"},
            return_timestamps=True
        )
        return result
    except Exception as e:
        logging.error(f"Pipeline transcription error: {e}")
        return None

def process_audio_pipeline(file_path):
    """Process large audio files using the pipeline model."""
    chunks = split_audio_pipeline(file_path)
    chunk_results = []
    transcriptions = []
    
    for chunk in chunks:
        result = transcribe_chunk_pipeline(chunk)
        if result:
            chunk_results.append(result)
            transcriptions.append(result.get('text', ''))
        os.remove(chunk)  # Clean up
    
    # Create segments from all chunks
    segments = create_segments_from_chunks(chunk_results)
    
    return " ".join(transcriptions), "en", segments

def transcribe_chunk_whisper(audio_chunk, chunk_path):
    """Transcribe a chunk using the standard Whisper model."""
    audio_chunk.export(chunk_path, format="wav")
    logging.info(f"Transcribing chunk with Whisper: {chunk_path}")
    result = whisper_model.transcribe(chunk_path)
    os.remove(chunk_path)
    return result['text'], result['language'], result['segments']

def process_audio_whisper_chunks(file_path):
    """Process audio in chunks using the standard Whisper model."""
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    transcription = ""
    language = None
    segments = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        chunk_path_template = "/tmp/temp_chunk_{}.wav"
        
        for i in range(0, total_length_ms, CHUNK_LENGTH_MS):
            chunk_start = i
            chunk_end = min(i + CHUNK_LENGTH_MS, total_length_ms)
            audio_chunk = audio[chunk_start:chunk_end]
            chunk_path = chunk_path_template.format(i)
            futures.append(executor.submit(transcribe_chunk_whisper, audio_chunk, chunk_path))
            
        for future in futures:
            text, lang, segs = future.result()
            transcription += text + " "
            if language is None:
                language = lang
            # Adjust segment timestamps based on their position in the full audio
            for seg in segs:
                segments.append(seg)
            
    return transcription.strip(), language, segments

def process_full_audio_whisper(file_path):
    """Process entire audio file using standard Whisper model."""
    result = whisper_model.transcribe(file_path)
    return result['text'], result['language'], result['segments']

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_token: str = Header(None)
):
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as audio_file:
        content = await file.read()
        audio_file.write(content)

    try:
        # Check file size
        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")

        # Choose processing method based on file size
        if file_size_mb > CHUNK_SIZE_MB:
            logging.info("Using pipeline model for large file...")
            transcription, language, segments = process_audio_pipeline(temp_file_path)
        else:
            logging.info("Using standard Whisper model...")
            if file_size_mb > 25:  # If file is between 10MB and 25MB, use chunking
                transcription, language, segments = process_audio_whisper_chunks(temp_file_path)
            else:
                transcription, language, segments = process_full_audio_whisper(temp_file_path)

        return JSONResponse(content={
            "results": [{
                "filename": file.filename,
                "transcript": {
                    "text": transcription,
                    "segments": segments,
                    "language": language
                }
            }]
        })

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
def read_root():
    return {"message": "Whisper Transcription API"}
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
import whisper
import os
import torch
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
import requests

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

# Initialize Whisper model
whisper_model = whisper.load_model("medium").to(device)

def split_audio_into_chunks(file_path, chunk_duration_ms=60000):
    """Split audio into chunks of fixed duration for processing."""
    logging.info("Splitting audio into chunks...")
    audio = AudioSegment.from_file(file_path)
    chunks = []
    
    # Ensure consistent chunk size
    for i in range(0, len(audio), chunk_duration_ms):
        chunk_end = min(i + chunk_duration_ms, len(audio))
        chunk = audio[i:chunk_end]
        
        if len(chunk) < chunk_duration_ms:
            logging.warning(f"Last chunk is smaller than the expected size: {len(chunk)} ms")
        
        temp_chunk = NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append(temp_chunk.name)
        
    return chunks

def transcribe_chunk_whisper(audio_chunk, chunk_path):
    """Transcribe a chunk using the standard Whisper model."""
    audio_chunk.export(chunk_path, format="wav")
    logging.info(f"Transcribing chunk with Whisper: {chunk_path}")
    
    try:
        result = whisper_model.transcribe(chunk_path)
        os.remove(chunk_path)
        return result['text'], result['language'], result['segments']
    except Exception as e:
        logging.error(f"Error transcribing chunk {chunk_path}: {e}")
        os.remove(chunk_path)  # Ensure temporary files are removed
        return "", "unknown", []  # Return empty values in case of failure

def process_audio_whisper_chunks(file_path):
    """Process audio in chunks using the standard Whisper model."""
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    transcription = ""
    language = None
    segments = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        chunk_path_template = "/tmp/temp_chunk_{}.wav"
        
        for i in range(0, total_length_ms, CHUNK_LENGTH_MS):
            chunk_start = i
            chunk_end = min(i + CHUNK_LENGTH_MS, total_length_ms)
            audio_chunk = audio[chunk_start:chunk_end]
            chunk_path = chunk_path_template.format(i)
            futures.append(executor.submit(transcribe_chunk_whisper, audio_chunk, chunk_path))
            
        for future in futures:
            try:
                text, lang, segs = future.result()
                transcription += text + " "
                if language is None:
                    language = lang
                # Adjust segment timestamps based on their position in the full audio
                for seg in segs:
                    segments.append(seg)
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
    
    return transcription.strip(), language, segments

def download_audio(url):
    """Download audio file from a URL."""
    try:
        logging.info(f"Downloading audio file from URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            temp_file_path = "/tmp/temp_audio_file"
            with open(temp_file_path, "wb") as f:
                f.write(response.content)
            return temp_file_path
        else:
            raise HTTPException(status_code=400, detail="Failed to download audio file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading audio file: {str(e)}")

@app.post("/transcribe/")
async def transcribe_audio(
    url: str,  # URL to download the audio file
    x_token: str = Header(None)
):
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Download the audio file from URL
    temp_file_path = download_audio(url)

    try:
        # Check file size
        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")

        # Choose processing method based on file size
        if file_size_mb > CHUNK_SIZE_MB:
            logging.info("Using pipeline model for large file...")
            transcription, language, segments = process_audio_whisper_chunks(temp_file_path)
        else:
            logging.info("Using standard Whisper model...")
            transcription, language, segments = process_audio_whisper_chunks(temp_file_path)

        return JSONResponse(content={
            "results": [{
                "filename": os.path.basename(temp_file_path),
                "transcript": {
                    "text": transcription,
                    "segments": segments,
                    "language": language
                }
            }]
        })

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/")
def read_root():
    return {"message": "Whisper Transcription API"}
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
import whisper
import os
import torch
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from tempfile import NamedTemporaryFile
import time

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
    
    # Initialize the pipeline with correct parameters
    pipeline_model = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-medium",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        return_timestamps=True  # Enable timestamp generation
    )

# Load models at startup
load_models()

def create_segments_from_chunks(chunks_results, chunk_duration_ms=180000):
    """Create segments from pipeline chunks with timestamps."""
    segments = []
    total_duration = 0
    
    for chunk_idx, chunk_result in enumerate(chunks_results):
        chunk_offset = chunk_idx * chunk_duration_ms / 1000  # Convert to seconds
        
        # If the chunk has chunks (timestamps), process them
        if 'chunks' in chunk_result:
            for chunk in chunk_result['chunks']:
                timestamp = chunk.get('timestamp', None)
                if timestamp:
                    start, end = timestamp
                    # Adjust timestamps based on chunk position
                    start = start + chunk_offset
                    end = end + chunk_offset
                    
                    segment = {
                        'id': len(segments),
                        'start': start,
                        'end': end,
                        'text': chunk['text'],
                        'tokens': [],  # Placeholder for token ids
                        'temperature': 0.0,
                        'avg_logprob': -1.0,
                        'compression_ratio': 1.0,
                        'no_speech_prob': 0.0
                    }
                    segments.append(segment)
                    total_duration = max(total_duration, end)
        else:
            # If no timestamps, create a single segment for the chunk
            segment = {
                'id': len(segments),
                'start': chunk_offset,
                'end': chunk_offset + (chunk_duration_ms / 1000),
                'text': chunk_result['text'],
                'tokens': [],
                'temperature': 0.0,
                'avg_logprob': -1.0,
                'compression_ratio': 1.0,
                'no_speech_prob': 0.0
            }
            segments.append(segment)
            total_duration = max(total_duration, segment['end'])
    
    return segments

def split_audio_pipeline(file_path, chunk_duration_ms=180000):
    """Split audio into fixed-size chunks for pipeline processing."""
    logging.info("Splitting audio into chunks for pipeline...")
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        temp_chunk = NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_chunk.name, format="wav")
        chunks.append(temp_chunk.name)
    return chunks

def transcribe_chunk_pipeline(chunk_path):
    """Transcribe a single chunk using the pipeline model."""
    try:
        logging.info(f"Transcribing chunk with pipeline: {chunk_path}")
        result = pipeline_model(
            chunk_path,
            generate_kwargs={"language": "en"},
            return_timestamps=True
        )
        return result
    except Exception as e:
        logging.error(f"Pipeline transcription error: {e}")
        return None

def process_audio_pipeline(file_path):
    """Process large audio files using the pipeline model."""
    chunks = split_audio_pipeline(file_path)
    chunk_results = []
    transcriptions = []
    
    for chunk in chunks:
        result = transcribe_chunk_pipeline(chunk)
        if result:
            chunk_results.append(result)
            transcriptions.append(result.get('text', ''))
        os.remove(chunk)  # Clean up
    
    # Create segments from all chunks
    segments = create_segments_from_chunks(chunk_results)
    
    return " ".join(transcriptions), "en", segments

def transcribe_chunk_whisper(audio_chunk, chunk_path):
    """Transcribe a chunk using the standard Whisper model."""
    audio_chunk.export(chunk_path, format="wav")
    logging.info(f"Transcribing chunk with Whisper: {chunk_path}")
    result = whisper_model.transcribe(chunk_path)
    os.remove(chunk_path)
    return result['text'], result['language'], result['segments']

def process_audio_whisper_chunks(file_path):
    """Process audio in chunks using the standard Whisper model."""
    audio = AudioSegment.from_file(file_path)
    total_length_ms = len(audio)
    transcription = ""
    language = None
    segments = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        chunk_path_template = "/tmp/temp_chunk_{}.wav"
        
        for i in range(0, total_length_ms, CHUNK_LENGTH_MS):
            chunk_start = i
            chunk_end = min(i + CHUNK_LENGTH_MS, total_length_ms)
            audio_chunk = audio[chunk_start:chunk_end]
            chunk_path = chunk_path_template.format(i)
            futures.append(executor.submit(transcribe_chunk_whisper, audio_chunk, chunk_path))
            
        for future in futures:
            text, lang, segs = future.result()
            transcription += text + " "
            if language is None:
                language = lang
            # Adjust segment timestamps based on their position in the full audio
            for seg in segs:
                segments.append(seg)
            
    return transcription.strip(), language, segments

def process_full_audio_whisper(file_path):
    """Process entire audio file using standard Whisper model."""
    result = whisper_model.transcribe(file_path)
    return result['text'], result['language'], result['segments']



@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    x_token: str = Header(None)
):
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as audio_file:
        content = await file.read()
        audio_file.write(content)

    try:
        # Check file size
        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
        logging.info(f"File size: {file_size_mb:.2f} MB")

        # Choose processing method based on file size
        if file_size_mb > CHUNK_SIZE_MB:
            logging.info("Using pipeline model for large file...")
            transcription, language, segments = process_audio_pipeline(temp_file_path)
        else:
            logging.info("Using standard Whisper model...")
            if file_size_mb > 25:  # If file is between 10MB and 25MB, use chunking
                transcription, language, segments = process_audio_whisper_chunks(temp_file_path)
            else:
                transcription, language, segments = process_full_audio_whisper(temp_file_path)

        # Create response with both detected language transcription and English transcription
        response = {
            "results": [{
                "filename": file.filename,
                "transcript": {
                    "text": transcription,
                    "language": language,
                    "segments": segments
                },
                "detected_language_transcription": {
                    "text": transcription,
                    "language": language
                }
            }]
        }

        return JSONResponse(content=response)

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



@app.get("/")
def read_root():
    return {"message": "Whisper Transcription API"}





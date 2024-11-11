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
from pydub import AudioSegment,silence
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

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

# Load models
whisper_model = whisper.load_model("medium").to(device)
whisperx_model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type="float32")
persian_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-persian")
persian_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-persian").to(device)
maltese_processor = Wav2Vec2Processor.from_pretrained("carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h")
maltese_model = Wav2Vec2ForCTC.from_pretrained("carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h").to(device)

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
    
    # Handle cases where no speech is detected
    if not result['text'].strip():
        logging.warning(f"No active speech detected in chunk: {chunk_path}")
        return "[No speech detected]"
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
    transcription = " ".join([segment['text'] for segment in result.get('segments', [])])
    return transcription

def transcribe_with_wav2vec(file_path, processor, model):
    """Transcribe using Wav2Vec2 for specific languages."""
    audio, rate = librosa.load(file_path, sr=16000)
    inputs = processor(audio, return_tensors="pt", padding=True).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_values=inputs["input_values"]).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

async def detect_language_for_chunk(audio_chunk_path):
    """Detect language for a given audio chunk using WhisperX or another model."""
    audio = whisperx.load_audio(audio_chunk_path)
    language_detection_result = whisperx_model.transcribe(audio, batch_size=4)
    return language_detection_result['language']

async def transcribe_chunk_based_on_language(audio_chunk, chunk_path):
    """Transcribe a chunk based on detected language."""
    # Save the audio chunk to a temporary file
    audio_chunk.export(chunk_path, format="wav")
    
    # Detect language for the chunk
    detected_language = await detect_language_for_chunk(chunk_path)
    logging.info(f"Detected language for chunk: {detected_language}")

    # Select the model based on the detected language
    if detected_language == 'en':  # English
        model = whisper_model
    elif detected_language == 'mt':  # Maltese
        model = maltese_model
    elif detected_language == 'fa':  # Persian (as an example for another language)
        model = persian_model
    else:
        model = whisper_model  # Default to Whisper for unknown languages
    
    # Transcribe the audio chunk
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: model.transcribe(chunk_path))
    
    os.remove(chunk_path)  # Clean up temporary file
    return result['text']

async def process_audio_with_multiple_languages(file_path):
    """Process the audio file in chunks and transcribe based on language."""
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
        tasks.append(transcribe_chunk_based_on_language(audio_chunk, chunk_path))

    results = await asyncio.gather(*tasks)
    
    # Filter out any "[No speech detected]" placeholders
    meaningful_results = [result for result in results if result != "[No speech detected]"]

    if not meaningful_results:
        return "No active speech detected in the entire audio."
    
    transcription = " ".join(meaningful_results)
    return transcription



def has_active_speech(file_path):
    """ Detects if the audio contains any active speech by checking for non-silent segments. """
    audio = AudioSegment.from_file(file_path)
    non_silent_segments = silence.detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-40)
    return len(non_silent_segments) > 0



@app.post("/filter_transcribe/")
async def transcribe_audio(file: UploadFile = File(...), x_token: str = Header(None)):
    """Main function to handle the uploaded audio and detect multiple languages."""
    # Validate token
    if x_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid token")

    # Save the uploaded audio file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    await save_audio_file(file, temp_file_path)

    # Check for active speech
    if not has_active_speech(temp_file_path):
        logging.info("No active speech found in audio; stopping process.")
        os.remove(temp_file_path)  # Clean up the temporary file
        return  # Stop the process without returning any response

    # Check cache before processing
    file_hash = get_file_hash(temp_file_path)
    cache_key = f"transcription_{file_hash}"
    cached_transcription = cache.get(cache_key)

    if cached_transcription:
        logging.info("Returning cached transcription.")
        os.remove(temp_file_path)  # Clean up the temporary file
        return JSONResponse(content={"transcription": cached_transcription})

    # Process the file and get the transcription
    transcription = await process_audio_with_multiple_languages(temp_file_path)

    # Cache the result
    cache[cache_key] = transcription

    # Clean up the temporary file
    os.remove(temp_file_path)

    # Return the transcription result
    return JSONResponse(content={"transcription": transcription})




@app.get("/")
def read_root():
    return {"message": "Hello World"}

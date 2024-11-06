import whisperx
import gc

# Set device to CPU
device = "cpu"
audio_file = "Malayalam.mp3"
batch_size = 4  # Reduced for CPU performance
compute_type = "float32"  # CPU compatibility

# 1. Load the Whisper model
model = whisperx.load_model("kurianbenoy/vegam-whisper-medium-ml", device, compute_type=compute_type)

# Load audio file
audio = whisperx.load_audio(audio_file)

# Perform transcription
result = model.transcribe(audio, batch_size=batch_size)
print("Transcription result (segments before alignment):")

segments = result['segments']

# Extract Malayalam text from segments
malayalam_text = [entry['text'] for entry in segments]

# Display the Malayalam words
for text in malayalam_text:
    print(text)
    

# # Free up memory by clearing the model if needed
gc.collect()





# # 2. Align Whisper output
# model_a, metadata = whisperx.load_align_model(language_code="ml", device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print("Alignment result (segments after alignment):")
# print(result["segments"])

# # Free up memory by clearing the alignment model if needed
# gc.collect()

# # 3. Assign Speaker Labels (if diarization is required)
# # Note: This step may be slow on CPU and requires sufficient RAM
# diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

# # Perform speaker diarization
# diarize_segments = diarize_model(audio)
# result = whisperx.assign_word_speakers(diarize_segments, result)

# # Print results with speaker IDs
# print("Speaker Diarization result:")
# print(diarize_segments)
# print("Final Transcription with speaker IDs:")
# print(result["segments"])  # Segments are now assigned speaker IDs

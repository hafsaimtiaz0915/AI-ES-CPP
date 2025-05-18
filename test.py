import os
import sys
import ffmpeg
import whisper
import torch
from transformers import pipeline
import numpy as np
from datetime import timedelta

# Ensure output encoding is UTF-8 (fix for Windows emoji/Unicode errors)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass  # For Python <3.7

def get_video_duration(video_path):
    """Get the duration of the video using ffmpeg"""
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])
    return duration

def extract_audio_chunk(video_path, start_time, chunk_duration, output_audio):
    """Extract a chunk of audio from the video"""
    print(f"[*] Extracting audio chunk from {start_time}s to {start_time + chunk_duration}s...")
    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, t=chunk_duration)
            .output(output_audio, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio chunk: {e.stderr.decode()}")
        return False

def transcribe_chunk(model, audio_path, start_time):
    """Transcribe a single audio chunk"""
    result = model.transcribe(
        audio_path,
        initial_prompt="This is a meeting transcript. Please maintain proper punctuation and capitalization.",
        fp16=False  # Explicitly disable FP16
    )
    timestamp = str(timedelta(seconds=int(start_time)))
    return f"[{timestamp}] {result['text']}\n"

def process_video_in_chunks(video_path, chunk_duration=300):  # 5 minutes chunks
    print("[*] Starting video processing...")
    
    if not os.path.exists(video_path):
        print(f"[!] File not found: {video_path}")
        return None, None
    
    # Get video duration
    total_duration = get_video_duration(video_path)
    print(f"[*] Total video duration: {total_duration:.2f} seconds")
    
    # Load Whisper model with explicit CPU and FP32 settings
    print("[*] Loading Whisper model...")
    model = whisper.load_model(
        "base",
        device="cpu",
        download_root=None,
        in_memory=True
    )
    # Force FP32
    model = model.float()
    
    # Initialize transcript
    full_transcript = ""
    temp_audio = "temp_chunk.wav"
    
    # Process video in chunks
    start_time = 0
    while start_time < total_duration:
        current_chunk_duration = min(chunk_duration, total_duration - start_time)
        
        print(f"\n[*] Processing chunk: {start_time:.1f}s to {start_time + current_chunk_duration:.1f}s")
        
        # Extract audio chunk
        if extract_audio_chunk(video_path, start_time, current_chunk_duration, temp_audio):
            # Transcribe chunk
            chunk_transcript = transcribe_chunk(model, temp_audio, start_time)
            full_transcript += chunk_transcript
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        
        start_time += chunk_duration
        print(f"[*] Progress: {min(100, (start_time/total_duration)*100):.1f}%")
    
    # Summarize the full transcript
    print("\n[*] Generating summary...")
    summary = summarize_text(full_transcript)
    
    return full_transcript, summary

def summarize_text(text, max_words=800):
    print("[*] Summarizing transcript with Transformers...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    words = text.split()
    
    # Process long text in chunks if needed
    if len(words) > max_words:
        summaries = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunk_summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(chunk_summary[0]['summary_text'])
        return " ".join(summaries)
    else:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

def save_output(transcript, summary, filename="transcript_summary.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPT:\n\n" + transcript + "\n\nSUMMARY:\n\n" + summary)
    print(f"[*] Transcript and summary saved to: {filename}")

def summarize_meeting(video_path):
    transcript, summary = process_video_in_chunks(video_path)
    if transcript and summary:
        save_output(transcript, summary)
        print("[✓] Done.")
    else:
        print("[!] Processing failed.")

# Replace this with your actual video file name
if __name__ == "__main__":
    video_path = "video.mp4"  # 🔁 Change this filename to your real video
    summarize_meeting(video_path)
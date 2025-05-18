import os
import sys
import ffmpeg
import whisper
from transformers import pipeline

# Ensure output encoding is UTF-8 (fix for Windows emoji/Unicode errors)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass  # For Python <3.7

def extract_audio(video_path, output_audio="temp_audio.wav"):
    print("[*] Extracting audio from video...")
    ffmpeg.input(video_path).output(output_audio).run(overwrite_output=True)
    return output_audio

def transcribe_audio(audio_path):
    print("[*] Transcribing audio with Whisper...")
    model = whisper.load_model("base")  # You can use 'small' or 'medium' for higher quality
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text, max_words=800):
    print("[*] Summarizing transcript with Transformers...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def save_output(transcript, summary, filename="transcript_summary.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPT:\n\n" + transcript + "\n\nSUMMARY:\n\n" + summary)
    print(f"[*] Transcript and summary saved to: {filename}")

def summarize_meeting(video_path):
    if not os.path.exists(video_path):
        print(f"[!] File not found: {video_path}")
        return

    audio_file = extract_audio(video_path)
    transcript = transcribe_audio(audio_file)
    summary = summarize_text(transcript)

    save_output(transcript, summary)

    # Optional: Cleanup
    if os.path.exists(audio_file):
        os.remove(audio_file)

    print("[✓] Done.")

# Replace this with your actual video file name
if __name__ == "__main__":
    video_path = "meeting_video.mp4"  # 🔁 Change this filename to your real video
    summarize_meeting(video_path)

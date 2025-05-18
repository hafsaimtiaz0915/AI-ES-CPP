# Video Transcription and Summarization

This Python script uses OpenAI's Whisper and Hugging Face's transformers to transcribe video content and generate summaries.

## Features

- Extracts audio from video files using FFmpeg
- Transcribes audio using OpenAI's Whisper
- Generates summaries using BART-large-CNN model
- Saves both transcript and summary to a text file

## Requirements

- Python 3.7+
- ffmpeg
- openai-whisper
- transformers
- ffmpeg-python

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv whisper-env
```
3. Activate the virtual environment:
```bash
# Windows
.\whisper-env\Scripts\activate
```
4. Install dependencies:
```bash
pip install ffmpeg-python openai-whisper transformers
```

## Usage

1. Place your video file in the project directory
2. Activate the virtual environment:
```powershell
# Windows PowerShell
.\whisper-env\Scripts\Activate.ps1

# Windows Command Prompt
.\whisper-env\Scripts\activate.bat
```

3. Install the required dependencies:
```powershell
pip install -r requirements.txt
```

4. Run the script with your video file:
```powershell
# Using the default video name (meeting_video.mp4)
python test.py

# Or specify a different video file by editing test.py
# Change the video_path variable at the bottom of test.py:
# video_path = "your_video_file.mp4"
```

The script will:
- Process the video in 5-minute chunks
- Show progress as it processes
- Generate timestamps in the transcript
- Create a summary of the content
- Save both transcript and summary in `transcript_summary.txt`

## Expected Output

The script will show progress indicators:
```
[*] Starting video processing...
[*] Total video duration: XXX seconds
[*] Loading Whisper model...
[*] Processing chunk: 0.0s to 300.0s
...
[✓] Done.
```

The output file `transcript_summary.txt` will contain:
- Full transcript with timestamps
- Summary of the video content

Run Command
cd "c:\Users\hafsa\Desktop\whisper-env" ; .\Scripts\Activate.ps1 ; python test.py
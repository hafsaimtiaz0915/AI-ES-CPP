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
2. Update the video_path in test.py
3. Run the script:
```bash
python test.py
```

The script will generate a `transcript_summary.txt` file containing both the full transcript and its summary.

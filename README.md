
# 🎥 Video Transcription and Summarization Tool

This Python-based solution enables **automatic transcription** and **summarization of video content** using state-of-the-art machine learning models. Leveraging **OpenAI's Whisper** for speech-to-text conversion and **Hugging Face's BART-large-CNN** for text summarization, the script seamlessly converts video files into insightful text summaries.

---
## ✨ Project Report

Full project report: [Google Docs Link](https://docs.google.com/document/d/1uH4otyxasE608TnfxS0YuzU77NniHZed5bbi-1HAeok/edit?usp=sharing)


## ✨ Youtube Link

Video Explanation + demo: [Youtube Link](https://youtu.be/VwW_Afq_0Yg)

## ✨ Key Features

* 🔊 **Audio Extraction**: Extracts audio directly from `.mp4` video files using **FFmpeg**.
* 📝 **Accurate Transcription**: Utilizes **OpenAI’s Whisper** for multilingual, high-accuracy speech recognition.
* 📄 **Summarization**: Generates coherent and concise summaries using the **BART-large-CNN** transformer model.
* 💾 **Auto Save**: Saves the full transcript and the summary to a neatly formatted text file.

---

## 🧰 Requirements

Ensure the following tools and libraries are installed in your environment:

* Python ≥ 3.7
* [FFmpeg](https://ffmpeg.org/download.html) (must be accessible via system PATH)
* Python Libraries:

  * `openai-whisper`
  * `transformers`
  * `ffmpeg-python`

---

## ⚙️ Installation Guide

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/video-transcription-tool.git
cd video-transcription-tool
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv whisper-env
```

Activate it:

```powershell
# PowerShell
.\whisper-env\Scripts\Activate.ps1
```

or

```cmd
# Command Prompt
.\whisper-env\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install ffmpeg-python openai-whisper transformers
```

Alternatively, use the requirements file:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Use

1. **Place your `.mp4` video file** in the project directory.

2. **Activate your virtual environment** (if not already activated):

```powershell
cd "C:\Users\YourName\Desktop\whisper-env"
.\Scripts\Activate.ps1
```

3. **Run the script**:

```bash
python test.py
```

> 🔁 You can also edit the video file name in `test.py`:
>
> ```python
> video_path = "your_video_file.mp4"
> ```

---

## 📌 Script Workflow

When executed, the script will:

* Read the video and divide it into 5-minute audio chunks.
* Transcribe each chunk using the Whisper model.
* Generate timestamps and log progress in the console.
* Summarize the transcript using the BART-large-CNN model.
* Save both the transcript and summary in `transcript_summary.txt`.

---

## 🖨️ Console Output Example

```
[*] Starting video processing...
[*] Total video duration: 935 seconds
[*] Loading Whisper model...
[*] Processing chunk: 0.0s to 300.0s
[*] Processing chunk: 300.0s to 600.0s
...
[✓] Transcription complete.
[✓] Generating summary...
[✓] Output saved to transcript_summary.txt
```

---

## 📄 Output Format

The file `transcript_summary.txt` will include:

```
=== TRANSCRIPT (with timestamps) ===
[00:00:01] Speaker: Welcome everyone...
...

=== SUMMARY ===
In this meeting, the team discussed project milestones, highlighted risks...
```

---

## 🏁 Run Command Summary

If you're running from PowerShell:

```powershell
cd "C:\Users\hafsa\Desktop\whisper-env"
.\Scripts\Activate.ps1
python test.py
```

---

## 👩‍💻 Contributors

This project was developed by students from **Batch 2022** as part of their academic work:

* **Zainab Furqan Ahmed** – CT-22067
* **Hafsa Imtiaz** – CT-22060
* **Sheeza Aslam** – CT-22064



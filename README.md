
# 🎥 Comparat## ✨ Key Features

### Core Functionality
* 🔊 **Audio Extraction**: Extracts audio directly from `.mp4` video files using **FFmpeg**
* 📝 **Accurate Transcription**: Utilizes **OpenAI's Whisper** for multilingual, high-accuracy speech recognition
* 📄 **Dual Summarization**: 
  - **Abstractive** summaries using **BART-large-CNN** transformer model
  - **Extractive** summaries using **TextRank** and **TF-IDF** algorithms

### Advanced NLP Features
* 🤖 **Question-Answering System**: Both structured (keyword-based) and unstructured (transformer-based) approaches
* 🏷️ **Named Entity Recognition**: Identifies and categorizes entities using **spaCy**
* 📊 **Topic Modeling**: Discovers themes using **Latent Dirichlet Allocation (LDA)**
* 🔍 **Comparative Analysis**: Side-by-side evaluation of structured vs unstructured methods

### Evaluation & Benchmarking
* 📈 **Performance Metrics**: ROUGE, BLEU scores, processing time, memory usage
* 🗃️ **Dataset Integration**: Support for CNN/DailyMail, SQuAD, SAMSum, and custom datasets
* 📋 **Comprehensive Reporting**: Automated evaluation reports with detailed comparisons

### User Interface
* 🖥️ **Enhanced GUI**: Multi-tab interface with comparison views and Q&A functionality
* 💻 **Command Line**: Automated batch processing with method selection
* 💾 **Smart Export**: Saves complete analysis including all method resultsudy of Structured and Unstructured NLP Models

This comprehensive Python-based solution enables **automatic transcription**, **summarization**, and **question-answering** for video content using both structured and unstructured NLP approaches. The system provides a comparative framework to evaluate traditional methods (TF-IDF, TextRank, LDA) against modern transformer-based models (Whisper, BART, T5, DistilBERT).

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

### System Requirements
* Python ≥ 3.7 (Python 3.8+ recommended)
* [FFmpeg](https://ffmpeg.org/download.html) (must be accessible via system PATH)
* Minimum 8GB RAM (16GB recommended for large models)
* GPU support optional but recommended for faster processing

### Core Libraries
* `openai-whisper` - Speech-to-text transcription
* `transformers` - Modern NLP models (BART, T5, DistilBERT)
* `torch` - PyTorch backend for neural networks
* `ffmpeg-python` - Audio/video processing

### Structured NLP Libraries
* `spacy` - Named Entity Recognition and text processing
* `scikit-learn` - TF-IDF vectorization and traditional ML
* `networkx` - Graph algorithms for TextRank
* `gensim` - Topic modeling with LDA

### Evaluation & Analysis
* `rouge-score` - ROUGE metrics for summarization evaluation
* `sacrebleu` - BLEU scores for text comparison
* `datasets` - Benchmark dataset loading (CNN/DailyMail, SQuAD, etc.)
* `psutil` - System performance monitoring

### GUI & Utilities
* `ttkthemes` - Enhanced GUI styling
* `pandas`, `numpy` - Data processing and analysis
* `pillow` - Image processing for GUI

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

Install all required packages:

```bash
pip install -r requirements.txt
```

Install spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### 4. Verify Installation

Test your setup with the comprehensive demo:

```bash
python demo.py
```

---

## ▶️ How to Use

### 🖥️ Graphical Interface (Recommended)

1. **Launch the enhanced GUI application**:

```bash
python app.py
```

2. **Use the comprehensive interface**:
   - **Transcription Tab**: Process video files with real-time progress
   - **Summary Tabs**: View both abstractive (BART) and extractive (TextRank) summaries
   - **Method Comparison**: Side-by-side analysis of structured vs unstructured approaches
   - **Q&A System**: Ask questions about transcribed content using both methods
   - **Evaluation**: Performance metrics and comparative analysis

### 💻 Command Line Interface

1. **Quick processing with default settings**:

```bash
python test.py
```

2. **Comprehensive demonstration**:

```bash
python demo.py
```

3. **Custom video file**:

```python
# Edit video_path in test.py
video_path = "your_video_file.mp4"
```

### � Advanced Features

**Question-Answering System**:
```python
from qa_system import QuestionAnsweringSystem

qa = QuestionAnsweringSystem()
qa.set_context("Your transcript text here...")
result = qa.answer_question("What was discussed?", method="both")
```

**Comparative Evaluation**:
```python
from evaluation_metrics import EvaluationMetrics

evaluator = EvaluationMetrics()
results = evaluator.comparative_evaluation(text, structured_func, unstructured_func)
evaluator.generate_evaluation_report(results, "report.txt")
```

**Dataset Integration**:
```python
from dataset_handler import DatasetHandler

handler = DatasetHandler()
handler.load_benchmark_datasets(sample_size=50)
samples = handler.get_evaluation_samples("cnn_dailymail", num_samples=10)
```

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



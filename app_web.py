"""
Flask Web Application for Audio Transcription & Summarization
Modern web interface replacing Tkinter - All model logic preserved
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import threading
import uuid
import time
from datetime import timedelta

# Lazy imports to speed up startup
whisper = None
torch = None
pipeline = None
ffmpeg = None

def lazy_import_models():
    """Import heavy libraries only when needed"""
    global whisper, torch, pipeline, ffmpeg
    if whisper is None:
        import whisper as w
        import torch as t
        from transformers import pipeline as p
        import ffmpeg as f
        whisper, torch, pipeline, ffmpeg = w, t, p, f

# Import existing modules - SAME LOGIC
ADVANCED_FEATURES_AVAILABLE = False
try:
    from qa_system import QuestionAnsweringSystem
    from evaluation_metrics import EvaluationMetrics
    from test import structured_summarization_pipeline, summarize_text
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"[!] Advanced features not available: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'audio-transcription-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage
whisper_models = {}
summarizer_model = None
processing_tasks = {}
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'wma', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_whisper_model(model_name='base'):
    """Load Whisper model - SAME LOGIC"""
    lazy_import_models()
    if model_name not in whisper_models:
        print(f"Loading Whisper {model_name} model...")
        whisper_models[model_name] = whisper.load_model(model_name)
    return whisper_models[model_name]

def load_summarizer():
    """Load BART summarizer - SAME LOGIC"""
    lazy_import_models()
    global summarizer_model
    if summarizer_model is None:
        print("Loading BART summarizer...")
        summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer_model

def get_video_duration(video_path):
    """Get video duration using ffmpeg - SAME LOGIC"""
    lazy_import_models()
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        return None

def extract_audio_chunk(video_path, start_time, chunk_duration, temp_name):
    """Extract audio chunk - SAME LOGIC"""
    lazy_import_models()
    try:
        temp_chunk = os.path.join(app.config['UPLOAD_FOLDER'], f"chunk_{temp_name}_{start_time}.wav")
        (
            ffmpeg
            .input(video_path, ss=start_time, t=chunk_duration)
            .output(temp_chunk, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return temp_chunk
    except Exception as e:
        print(f"Error extracting chunk: {e}")
        return None

def process_audio_video(task_id, filepath, model_name, chunk_size):
    """Background processing - EXACT SAME LOGIC as Tkinter app.py"""
    task = processing_tasks[task_id]
    
    try:
        # Get duration
        duration = get_video_duration(filepath)
        if not duration:
            task['status'] = 'error'
            task['error'] = 'Could not determine video duration'
            return
        
        # Load models
        task['status'] = 'loading_models'
        task['progress'] = 0
        whisper_model = load_whisper_model(model_name)
        summarizer = load_summarizer()
        
        # Process in chunks - SAME LOGIC
        full_transcript = ""
        start_time = 0
        chunk_count = 0
        total_chunks = int(duration / chunk_size) + 1
        
        while start_time < duration and not task.get('cancelled'):
            current_chunk = min(chunk_size, duration - start_time)
            progress = int((start_time / duration) * 70)
            
            task['progress'] = progress
            task['status'] = 'transcribing'
            task['step'] = f'Chunk {chunk_count + 1}/{total_chunks}: {timedelta(seconds=int(start_time))}'
            
            # Extract and transcribe
            temp_chunk = extract_audio_chunk(filepath, start_time, current_chunk, task_id)
            if temp_chunk:
                try:
                    result = whisper_model.transcribe(temp_chunk)
                    timestamp = str(timedelta(seconds=int(start_time)))
                    chunk_text = f"[{timestamp}] {result['text']}\n\n"
                    
                    task['transcript'] += chunk_text
                    full_transcript += chunk_text
                    
                    # Cleanup temp file
                    if os.path.exists(temp_chunk):
                        os.remove(temp_chunk)
                except Exception as e:
                    task['error'] = f"Error transcribing chunk: {str(e)}"
            
            start_time += chunk_size
            chunk_count += 1
        
        if task.get('cancelled'):
            task['status'] = 'cancelled'
            return
        
        # Generate summaries - SAME LOGIC
        if full_transcript:
            task['progress'] = 80
            task['status'] = 'summarizing'
            task['step'] = 'Creating abstractive summary...'
            
            # Abstractive summary (BART)
            chunks = [full_transcript[i:i + 1000] for i in range(0, len(full_transcript), 1000)]
            summary_texts = []
            
            for i, chunk in enumerate(chunks):
                if task.get('cancelled'):
                    break
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                summary_texts.append(summary[0]['summary_text'])
                task['progress'] = int(80 + (i / len(chunks)) * 10)
            
            task['abstractive_summary'] = " ".join(summary_texts)
            
            # Extractive summary if available
            if ADVANCED_FEATURES_AVAILABLE:
                task['progress'] = 90
                task['step'] = 'Creating extractive summary...'
                try:
                    structured_analysis = structured_summarization_pipeline(full_transcript)
                    task['extractive_summary'] = structured_analysis.get('extractive_summary', '')
                    task['entities'] = structured_analysis.get('entities', [])[:10]
                    task['topics'] = structured_analysis.get('topics', [])
                except Exception as e:
                    print(f"Error generating extractive summary: {e}")
        
        task['progress'] = 100
        task['status'] = 'completed'
        task['step'] = 'Done!'
        
    except Exception as e:
        task['status'] = 'error'
        task['error'] = str(e)
    finally:
        # Cleanup uploaded file after processing
        time.sleep(2)  # Brief delay before cleanup
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/')
def index():
    return render_template('index.html', advanced_features=ADVANCED_FEATURES_AVAILABLE)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save file
    task_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(filepath)
    
    # Get parameters
    model_name = request.form.get('model', 'base')
    chunk_size = int(request.form.get('chunk_size', 30))
    
    # Initialize task
    processing_tasks[task_id] = {
        'id': task_id,
        'filename': filename,
        'status': 'queued',
        'progress': 0,
        'step': 'Starting...',
        'transcript': '',
        'abstractive_summary': '',
        'extractive_summary': '',
        'entities': [],
        'topics': [],
        'error': None,
        'cancelled': False
    }
    
    # Start background processing
    thread = threading.Thread(target=process_audio_video, args=(task_id, filepath, model_name, chunk_size))
    thread.daemon = True
    thread.start()
    
    return jsonify({'task_id': task_id, 'filename': filename}), 200

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'step': task['step'],
        'transcript': task['transcript'],
        'abstractive_summary': task['abstractive_summary'],
        'extractive_summary': task['extractive_summary'],
        'entities': task['entities'],
        'topics': task['topics'],
        'error': task['error']
    }), 200

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel(task_id):
    """Cancel processing"""
    if task_id in processing_tasks:
        processing_tasks[task_id]['cancelled'] = True
        return jsonify({'message': 'Cancelled'}), 200
    return jsonify({'error': 'Task not found'}), 404

@app.route('/qa', methods=['POST'])
def qa_answer():
    """Q&A endpoint"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return jsonify({'error': 'QA not available'}), 503
    
    data = request.json
    question = data.get('question', '')
    task_id = data.get('task_id', '')
    method = data.get('method', 'both')
    
    if not question or task_id not in processing_tasks:
        return jsonify({'error': 'Invalid request'}), 400
    
    context = processing_tasks[task_id]['transcript']
    if not context:
        return jsonify({'error': 'No transcript available'}), 400
    
    try:
        qa_system = QuestionAnsweringSystem()
        qa_system.set_context(context)
        results = qa_system.answer_question(question, method)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Method comparison endpoint"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return jsonify({'error': 'Comparison not available'}), 503
    
    data = request.json
    context = data.get('context', '')
    
    if not context:
        return jsonify({'error': 'No context'}), 400
    
    results = {}
    try:
        structured_analysis = structured_summarization_pipeline(context)
        results['structured'] = structured_analysis
        
        summary = summarize_text(context)
        results['unstructured'] = {'summary': summary}
        
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluation endpoint"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return jsonify({'error': 'Evaluation not available'}), 503
    
    data = request.json
    context = data.get('context', '')
    
    if not context:
        return jsonify({'error': 'No context'}), 400
    
    try:
        evaluator = EvaluationMetrics()
        results = evaluator.comparative_evaluation(context, structured_summarization_pipeline, summarize_text)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Audio Transcription Web App Starting...")
    print("="*60)
    print(f"📡 Access at: http://localhost:5000")
    print(f"⚡ Advanced Features: {'✅ Available' if ADVANCED_FEATURES_AVAILABLE else '❌ Not Available'}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

"""
Flask Web Application for Audio Transcription & Summarization
Replaces Tkinter GUI with modern web interface - keeps all model logic intact
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
import os
import whisper
import torch
from transformers import pipeline
import ffmpeg
from datetime import timedelta
import threading
import queue
import uuid
from werkzeug.utils import secure_filename

# Import existing modules (same logic as before)
try:
    from qa_system import QuestionAnsweringSystem
    from evaluation_metrics import EvaluationMetrics
    from test import structured_summarization_pipeline, summarize_text
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"[!] Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global models (lazy loaded)
whisper_models = {}
summarizer = None
qa_system = None if not ADVANCED_FEATURES_AVAILABLE else QuestionAnsweringSystem()
evaluator = None if not ADVANCED_FEATURES_AVAILABLE else EvaluationMetrics()

# Active processing tasks
active_tasks = {}

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'wma', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_whisper_model(model_name='base'):
    """Lazy load Whisper models"""
    if model_name not in whisper_models:
        socketio.emit('status', {'message': f'Loading Whisper {model_name} model...', 'progress': 0})
        whisper_models[model_name] = whisper.load_model(model_name)
    return whisper_models[model_name]

def get_summarizer():
    """Lazy load summarizer"""
    global summarizer
    if summarizer is None:
        socketio.emit('status', {'message': 'Loading BART summarizer...', 'progress': 5})
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def get_video_duration(video_path):
    """Get duration using ffmpeg"""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        return None

def extract_audio_chunk(video_path, start_time, chunk_duration, task_id):
    """Extract audio chunk using ffmpeg"""
    try:
        temp_chunk = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_chunk_{task_id}_{start_time}.wav")
        (
            ffmpeg
            .input(video_path, ss=start_time, t=chunk_duration)
            .output(temp_chunk, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return temp_chunk
    except Exception as e:
        return None

def process_file_task(file_path, model_name, chunk_size, task_id):
    """Background task for processing audio/video file - SAME LOGIC as Tkinter version"""
    try:
        # Get duration
        duration = get_video_duration(file_path)
        if not duration:
            socketio.emit('error', {'message': 'Could not get video duration'}, room=task_id)
            return
        
        # Load models
        whisper_model = get_whisper_model(model_name)
        summarizer_model = get_summarizer()
        
        # Process in chunks (SAME LOGIC)
        full_transcript = ""
        start_time = 0
        chunk_count = 0
        total_chunks = int(duration / chunk_size) + 1
        
        while start_time < duration:
            if task_id not in active_tasks or active_tasks[task_id].get('cancelled'):
                break
                
            current_chunk = min(chunk_size, duration - start_time)
            progress = int((start_time / duration) * 70)
            
            socketio.emit('progress', {
                'progress': progress,
                'status': f'Processing chunk {chunk_count + 1}/{total_chunks}',
                'step': f'Transcribing {timedelta(seconds=int(start_time))} to {timedelta(seconds=int(start_time + current_chunk))}'
            }, room=task_id)
            
            # Extract and transcribe chunk
            temp_chunk = extract_audio_chunk(file_path, start_time, current_chunk, task_id)
            if temp_chunk:
                try:
                    result = whisper_model.transcribe(temp_chunk)
                    timestamp = str(timedelta(seconds=int(start_time)))
                    chunk_text = f"[{timestamp}] {result['text']}\n\n"
                    
                    # Send transcript chunk to client
                    socketio.emit('transcript_chunk', {'text': chunk_text}, room=task_id)
                    full_transcript += chunk_text
                    
                    # Clean up temp file
                    os.remove(temp_chunk)
                except Exception as e:
                    socketio.emit('error', {'message': f'Error processing chunk: {str(e)}'}, room=task_id)
            
            start_time += chunk_size
            chunk_count += 1
        
        if task_id not in active_tasks or active_tasks[task_id].get('cancelled'):
            socketio.emit('cancelled', {}, room=task_id)
            return
        
        # Generate summaries (SAME LOGIC)
        if full_transcript:
            socketio.emit('progress', {
                'progress': 80,
                'status': 'Generating summaries...',
                'step': 'Creating abstractive summary...'
            }, room=task_id)
            
            # Abstractive summary (BART)
            chunks = [full_transcript[i:i + 1000] for i in range(0, len(full_transcript), 1000)]
            summary_texts = []
            
            for i, chunk in enumerate(chunks):
                if task_id not in active_tasks or active_tasks[task_id].get('cancelled'):
                    break
                summary = summarizer_model(chunk, max_length=150, min_length=30, do_sample=False)
                summary_texts.append(summary[0]['summary_text'])
                progress = int(80 + (i / len(chunks)) * 10)
                socketio.emit('progress', {
                    'progress': progress,
                    'step': f'Abstractive summary part {i+1}/{len(chunks)}'
                }, room=task_id)
            
            final_summary = " ".join(summary_texts)
            socketio.emit('abstractive_summary', {'text': final_summary}, room=task_id)
            
            # Extractive summary (TextRank) if available
            if ADVANCED_FEATURES_AVAILABLE:
                socketio.emit('progress', {
                    'progress': 90,
                    'step': 'Creating extractive summary...'
                }, room=task_id)
                try:
                    structured_analysis = structured_summarization_pipeline(full_transcript)
                    extractive_summary = structured_analysis.get('extractive_summary', 'No extractive summary available')
                    socketio.emit('extractive_summary', {'text': extractive_summary}, room=task_id)
                    
                    # Store for QA system
                    if qa_system:
                        qa_system.set_context(full_transcript)
                        active_tasks[task_id]['transcript'] = full_transcript
                except Exception as e:
                    print(f"Error generating extractive summary: {e}")
        
        socketio.emit('complete', {
            'progress': 100,
            'status': 'Processing completed',
            'step': 'Done'
        }, room=task_id)
        
    except Exception as e:
        socketio.emit('error', {'message': str(e)}, room=task_id)
    finally:
        # Cleanup
        if task_id in active_tasks:
            del active_tasks[task_id]

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html', advanced_features=ADVANCED_FEATURES_AVAILABLE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(filepath)
        
        # Get processing parameters
        model_name = request.form.get('model', 'base')
        chunk_size = int(request.form.get('chunk_size', 30))
        
        # Store task info
        active_tasks[task_id] = {
            'filename': filename,
            'filepath': filepath,
            'cancelled': False,
            'transcript': ''
        }
        
        # Start background processing
        thread = threading.Thread(target=process_file_task, args=(filepath, model_name, chunk_size, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'filename': filename}), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel processing task"""
    if task_id in active_tasks:
        active_tasks[task_id]['cancelled'] = True
        return jsonify({'message': 'Task cancelled'}), 200
    return jsonify({'error': 'Task not found'}), 404

@app.route('/qa', methods=['POST'])
def answer_question():
    """Handle Q&A requests"""
    if not ADVANCED_FEATURES_AVAILABLE or not qa_system:
        return jsonify({'error': 'QA system not available'}), 503
    
    data = request.json
    question = data.get('question', '')
    method = data.get('method', 'both')
    task_id = data.get('task_id', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if task_id not in active_tasks or not active_tasks[task_id].get('transcript'):
        return jsonify({'error': 'No transcript available'}), 400
    
    try:
        results = qa_system.answer_question(question, method)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_methods():
    """Generate method comparison"""
    if not ADVANCED_FEATURES_AVAILABLE:
        return jsonify({'error': 'Comparison not available'}), 503
    
    data = request.json
    context = data.get('context', '')
    enable_structured = data.get('structured', True)
    enable_unstructured = data.get('unstructured', True)
    
    if not context:
        return jsonify({'error': 'No context provided'}), 400
    
    results = {}
    
    if enable_structured:
        try:
            structured_analysis = structured_summarization_pipeline(context)
            results['structured'] = {
                'summary': structured_analysis.get('extractive_summary', ''),
                'entities': structured_analysis.get('entities', [])[:10],
                'topics': structured_analysis.get('topics', [])
            }
        except Exception as e:
            results['structured'] = {'error': str(e)}
    
    if enable_unstructured:
        try:
            summary = summarize_text(context)
            results['unstructured'] = {
                'summary': summary,
                'model': 'facebook/bart-large-cnn',
                'input_length': len(context.split()),
                'summary_length': len(summary.split())
            }
        except Exception as e:
            results['unstructured'] = {'error': str(e)}
    
    return jsonify(results), 200

@app.route('/evaluate', methods=['POST'])
def run_evaluation():
    """Run comparative evaluation"""
    if not ADVANCED_FEATURES_AVAILABLE or not evaluator:
        return jsonify({'error': 'Evaluation not available'}), 503
    
    data = request.json
    context = data.get('context', '')
    
    if not context:
        return jsonify({'error': 'No context provided'}), 400
    
    try:
        results = evaluator.comparative_evaluation(
            context,
            structured_summarization_pipeline,
            summarize_text
        )
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('join')
def on_join(data):
    """Join a task room for updates"""
    task_id = data['task_id']
    socketio.server.enter_room(request.sid, task_id)
    emit('joined', {'task_id': task_id})

if __name__ == '__main__':
    print("🚀 Starting Audio Transcription Web App...")
    print("📡 Server will be available at: http://localhost:5000")
    print("⚡ Web interface provides better performance than Tkinter GUI")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

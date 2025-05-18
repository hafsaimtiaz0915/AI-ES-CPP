import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import whisper
import threading
import torch
import queue
from transformers import pipeline
import time
import ffmpeg
from datetime import timedelta
from ttkthemes import ThemedTk

class AudioTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription & Summarization Tool")
        self.root.geometry("1200x800")
        
        # Apply modern theme and styling
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Helvetica", 10))
        style.configure("Status.TLabel", font=("Helvetica", 9))
        
        # Configure custom button styles
        style.configure("Accent.TButton",
                      padding=10,
                      font=("Helvetica", 10, "bold"))
        
        style.configure("Action.TButton",
                      padding=8,
                      font=("Helvetica", 10))
                      
        # Configure progress bar
        style.configure("Horizontal.TProgressbar",
                      troughcolor='#E0E0E0',
                      background='#4CAF50',
                      thickness=15)

        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Create main frame with improved layout
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, relief="groove", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Title
        ttk.Label(left_panel, text="Audio Transcription", style="Title.TLabel").pack(pady=(0,10))
        
        # File selection with improved UI
        file_frame = ttk.LabelFrame(left_panel, text="Select Input File", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(file_frame, text="Browse", command=self.browse_file, style="Action.TButton").pack(side=tk.RIGHT, padx=5)
        
        # Settings frame with improved layout
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Model selection with better layout
        ttk.Label(settings_frame, text="Model Quality:", style="Subtitle.TLabel").grid(row=0, column=0, sticky=tk.W, padx=5, pady=(0,5))
        self.model_var = tk.StringVar(value="base")
        models = [
            ("Fast (Base)", "base", "Quickest, good for short clips"),
            ("Better (Small)", "small", "Balance of speed and accuracy"),
            ("Best (Medium)", "medium", "Most accurate, slower processing")
        ]
        
        for i, (text, value, tooltip) in enumerate(models):
            frame = ttk.Frame(settings_frame)
            frame.grid(row=i+1, column=0, sticky=tk.W, padx=5)
            radio = ttk.Radiobutton(frame, text=text, variable=self.model_var, value=value)
            radio.pack(side=tk.LEFT)
            ttk.Label(frame, text=f"({tooltip})", style="Status.TLabel", foreground='gray').pack(side=tk.LEFT, padx=5)
        
        # Chunk size setting with better layout
        ttk.Label(settings_frame, text="Processing Chunk Size:", style="Subtitle.TLabel").grid(row=4, column=0, sticky=tk.W, padx=5, pady=(10,5))
        self.chunk_size_var = tk.StringVar(value="30")
        chunk_frame = ttk.Frame(settings_frame)
        chunk_frame.grid(row=5, column=0, sticky=tk.W, padx=5)
        chunk_sizes = [
            ("30s", "30"),
            ("1min", "60"),
            ("2min", "120"),
            ("5min", "300")
        ]
        for text, value in chunk_sizes:
            ttk.Radiobutton(chunk_frame, text=text, variable=self.chunk_size_var, value=value).pack(side=tk.LEFT, padx=5)
        
        # Progress section with improved visuals
        progress_frame = ttk.LabelFrame(left_panel, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, mode='determinate',
                                          style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.step_label = ttk.Label(progress_frame, text="Ready to start", 
                                   style="Status.TLabel", wraplength=250)
        self.step_label.pack(fill=tk.X, padx=5)
        
        # Control buttons with improved layout
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.transcribe_button = ttk.Button(button_frame, text="▶ Start", 
                                          command=self.start_processing, 
                                          style="Accent.TButton", width=12)
        self.transcribe_button.pack(side=tk.LEFT, padx=2)
        
        self.cancel_button = ttk.Button(button_frame, text="⬛ Stop", 
                                      command=self.cancel_processing, 
                                      style="Action.TButton", width=12,
                                      state='disabled')
        self.cancel_button.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="🗑 Clear", 
                  command=self.clear_output, 
                  style="Action.TButton", width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="💾 Save", 
                  command=self.save_all, 
                  style="Action.TButton", width=12).pack(side=tk.LEFT, padx=2)

        # Right panel with notebook - improved styling
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Transcription tab
        transcription_frame = ttk.Frame(notebook, padding="5")
        notebook.add(transcription_frame, text=" 📝 Transcription ")
        
        self.output_text = ScrolledText(transcription_frame, wrap=tk.WORD, 
                                      height=25, font=("Consolas", 10),
                                      bg="#FFFFFF", fg="#000000")
        self.output_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(notebook, padding="5")
        notebook.add(summary_frame, text=" 📋 Summary ")
        
        self.summary_text = ScrolledText(summary_frame, wrap=tk.WORD, 
                                       height=25, font=("Consolas", 10),
                                       bg="#FFFFFF", fg="#000000")
        self.summary_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Status bar with improved styling
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                                  style="Status.TLabel", relief=tk.SUNKEN,
                                  padding=(5, 2))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Initialize models and state
        self.whisper_model = None
        self.summarizer = None
        self.processing = False
        self.cancelled = False
        
        # Start queue processing
        self.process_queue()

    def get_video_duration(self, video_path):
        """Get the duration of the video using ffmpeg"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            messagebox.showerror("Error", f"Could not get video duration: {str(e)}")
            return None

    def extract_audio_chunk(self, video_path, start_time, chunk_duration):
        """Extract a chunk of audio from the video"""
        try:
            temp_chunk = f"temp_chunk_{start_time}.wav"
            (
                ffmpeg
                .input(video_path, ss=start_time, t=chunk_duration)
                .output(temp_chunk, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return temp_chunk
        except Exception as e:
            self.update_ui(status=f"Error extracting audio chunk: {str(e)}")
            return None

    def process_file(self):
        try:
            file_path = self.file_path.get()
            chunk_duration = int(self.chunk_size_var.get())
            
            # Get total duration
            duration = self.get_video_duration(file_path)
            if not duration:
                return
                
            # Load models if needed
            if not self.whisper_model:
                self.update_ui(status="Loading Whisper model...", progress=0, step="Initializing...")
                self.whisper_model = whisper.load_model(self.model_var.get())
            
            if not self.summarizer:
                self.update_ui(status="Loading summarizer...", progress=5, step="Initializing summarizer...")
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            # Process in chunks
            full_transcript = ""
            start_time = 0
            chunk_count = 0
            total_chunks = int(duration / chunk_duration) + 1
            
            while start_time < duration and not self.cancelled:
                current_chunk = min(chunk_duration, duration - start_time)
                progress = (start_time / duration) * 70  # Use 70% of progress bar for transcription
                
                self.update_ui(
                    status=f"Processing chunk {chunk_count + 1}/{total_chunks}",
                    progress=progress,
                    step=f"Transcribing {timedelta(seconds=int(start_time))} to {timedelta(seconds=int(start_time + current_chunk))}"
                )
                
                # Extract and transcribe chunk
                temp_chunk = self.extract_audio_chunk(file_path, start_time, current_chunk)
                if temp_chunk:
                    try:
                        result = self.whisper_model.transcribe(temp_chunk)
                        timestamp = str(timedelta(seconds=int(start_time)))
                        chunk_text = f"[{timestamp}] {result['text']}\n"
                        
                        # Update transcription immediately
                        self.output_text.insert(tk.END, chunk_text)
                        self.output_text.see(tk.END)
                        full_transcript += chunk_text
                        
                        # Clean up temp file
                        os.remove(temp_chunk)
                    except Exception as e:
                        self.update_ui(status=f"Error processing chunk: {str(e)}")
                
                start_time += chunk_duration
                chunk_count += 1
            
            if self.cancelled:
                raise Exception("Processing cancelled by user")
            
            # Generate summary
            if full_transcript:
                self.update_ui(progress=80, status="Generating summary...", step="Creating summary...")
                chunks = [full_transcript[i:i + 1000] for i in range(0, len(full_transcript), 1000)]
                summary_texts = []
                
                for i, chunk in enumerate(chunks):
                    if self.cancelled:
                        break
                    summary = self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summary_texts.append(summary[0]['summary_text'])
                    self.update_ui(
                        progress=80 + (i / len(chunks)) * 20,
                        step=f"Summarizing part {i+1}/{len(chunks)}"
                    )
                
                final_summary = " ".join(summary_texts)
                self.summary_text.delete(1.0, tk.END)
                self.summary_text.insert(tk.END, final_summary)
            
            self.update_ui(progress=100, status="Processing completed", step="Done")
            
        except Exception as e:
            if not self.cancelled:
                messagebox.showerror("Error", str(e))
                self.update_ui(status="Processing failed", step="Error occurred")
        finally:
            self.processing = False
            self.transcribe_button.configure(state='normal')
            self.cancel_button.configure(state='disabled')
            
            # Clean up any remaining temp files
            for file in os.listdir():
                if file.startswith("temp_chunk_") and file.endswith(".wav"):
                    try:
                        os.remove(file)
                    except:
                        pass

    def process_queue(self):
        """Process messages from the queue to update the UI"""
        try:
            while True:
                msg = self.queue.get_nowait()
                if isinstance(msg, dict):
                    if 'progress' in msg:
                        self.progress_var.set(msg['progress'])
                    if 'status' in msg:
                        self.status_var.set(msg['status'])
                    if 'step' in msg:
                        self.step_label.config(text=msg['step'])
                    if 'output' in msg:
                        self.output_text.insert(tk.END, msg['output'])
                        self.output_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            # Schedule the next queue check
            self.root.after(100, self.process_queue)

    def update_ui(self, **kwargs):
        """Send updates to the UI thread"""
        self.queue.put(kwargs)

    def start_processing(self):
        if not self.file_path.get():
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.cancelled = False
        self.transcribe_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.clear_output()
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_file, daemon=True).start()

    def cancel_processing(self):
        self.cancelled = True
        self.update_ui(status="Cancelling...", step="Stopping processes...")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio/Video files", "*.mp3 *.mp4 *.wav *.m4a *.wma"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            self.update_ui(status=f"Selected: {os.path.basename(file_path)}")

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_var.set("")
        self.step_label.config(text="")

    def save_all(self):
        if not self.output_text.get(1.0, tk.END).strip() and not self.summary_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No content to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=== TRANSCRIPTION ===\n\n")
                    f.write(self.output_text.get(1.0, tk.END))
                    f.write("\n\n=== SUMMARY ===\n\n")
                    f.write(self.summary_text.get(1.0, tk.END))
                self.status_var.set(f"Saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # You can try different themes like 'arc', 'equilux', 'breeze'
    root.configure(bg='#f0f0f0')  # Light gray background
    app = AudioTranscriptionApp(root)
    root.mainloop()

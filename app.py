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

class AutoScrollbar(ttk.Scrollbar):
    """Scrollbar that automatically hides when not needed"""
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

class AudioTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription & Summarization")
        self.root.geometry("1500x800")
        
        # Set color scheme based on reference image
        self.colors = {
            "primary": "#1a73e8",      # Blue for primary actions
            "secondary": "#f0f6ff",    # Light blue for backgrounds
            "accent": "#ff5252",       # Red for accent elements
            "text_primary": "#202124", # Dark gray for main text
            "text_secondary": "#5f6368", # Medium gray for secondary text
            "background": "#ffffff",   # White background
            "input_bg": "#f8f9fa"      # Light gray for input backgrounds
        }
          # Main container for content
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with scrollbar for entire content
        self.canvas = tk.Canvas(main_container, bg=self.colors["background"])
        self.scrollbar = AutoScrollbar(main_container, orient="vertical", command=self.canvas.yview)
        
        # Create the main scrollable frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Make the frame expand to fill canvas width
        self.canvas.bind('<Configure>', self._configure_canvas)
        
        # Create a window inside the canvas to hold the scrollable frame
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Grid layout for canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(0, weight=1)        # Configure grid weights
        self.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # Pack canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Apply modern theme and styling
        style = ttk.Style()
        style.configure("TFrame", background=self.colors["background"])
        style.configure("Title.TLabel", 
                      font=("Segoe UI", 24, "bold"), 
                      foreground=self.colors["text_primary"],
                      background=self.colors["background"])
        
        style.configure("Subtitle.TLabel", 
                      font=("Segoe UI", 12),
                      foreground=self.colors["text_primary"],
                      background=self.colors["background"])
        
        style.configure("Status.TLabel", 
                      font=("Segoe UI", 10),
                      foreground=self.colors["text_secondary"],
                      background=self.colors["background"])
        
        # Configure custom button styles with proper text color
        style.configure("Primary.TButton",
                      font=("Segoe UI", 11, "bold"),
                      background=self.colors["primary"],
                      foreground="white")
        
        style.configure("Secondary.TButton",
                      font=("Segoe UI", 11),
                      background=self.colors["background"])
        
        style.map("Primary.TButton",
                background=[('active', '#0b5cbe'), ('pressed', '#0b5cbe')],
                foreground=[('active', 'white'), ('pressed', 'white'), ('!disabled', 'white')],
                relief=[('pressed', 'sunken')])
        
        # Configure progress bar
        style.configure("Horizontal.TProgressbar",
                      troughcolor='#E0E0E0',
                      background=self.colors["primary"],
                      thickness=8)

        # Frame for the hero section (now inside scrollable_frame)
        hero_frame = ttk.Frame(self.scrollable_frame, padding="20", style="TFrame")
        hero_frame.pack(fill=tk.X)
        
        # App title in hero section
        ttk.Label(hero_frame, 
                text="Audio Transcription Tool", 
                style="Title.TLabel").pack(pady=(10, 5))
        
        ttk.Label(hero_frame, 
                text="Convert speech to text, summarize content, and save your results with confidence.", 
                style="Subtitle.TLabel").pack(pady=(0, 20))

        # Queue for thread communication
        self.queue = queue.Queue()
          # Create main content area with improved layout
        main_content = ttk.Frame(self.scrollable_frame, padding="20", style="TFrame")
        main_content.pack(fill=tk.BOTH, expand=True)
        main_content.columnconfigure(1, weight=3)
        main_content.columnconfigure(0, weight=1)
        main_content.rowconfigure(0, weight=1)
        
        # Left panel for controls with rounded corners and shadow effect
        left_panel = ttk.Frame(main_content, padding="20", style="TFrame")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))
        
        # File selection with improved UI
        file_section = ttk.Frame(left_panel, style="TFrame")
        file_section.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(file_section, 
                text="Select Audio or Video File", 
                font=("Segoe UI", 14, "bold"),
                foreground=self.colors["text_primary"],
                background=self.colors["background"]).pack(anchor=tk.W, pady=(0, 10))
        
        file_frame = ttk.Frame(file_section, style="TFrame")
        file_frame.pack(fill=tk.X)
        
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(file_frame, 
                             textvariable=self.file_path,
                             font=("Segoe UI", 10))
        file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10))
        
        browse_button = ttk.Button(file_frame, 
                                 text="Browse",
                                 command=self.browse_file,
                                 style="Secondary.TButton")
        browse_button.pack(side=tk.RIGHT)
        
        # Add separator
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Settings frame with improved layout
        settings_frame = ttk.Frame(left_panel, style="TFrame")
        settings_frame.pack(fill=tk.X)
        
        ttk.Label(settings_frame, 
                text="Model Settings", 
                font=("Segoe UI", 14, "bold"),
                foreground=self.colors["text_primary"],
                background=self.colors["background"]).pack(anchor=tk.W, pady=(0, 15))
        
        # Model quality section
        quality_frame = ttk.Frame(settings_frame, style="TFrame")
        quality_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(quality_frame, 
                text="Transcription Quality:", 
                font=("Segoe UI", 11),
                foreground=self.colors["text_primary"],
                background=self.colors["background"]).pack(anchor=tk.W, pady=(0, 5))
        
        self.model_var = tk.StringVar(value="base")
        models = [
            ("Fast (Base)", "base", "Quickest, good for short clips"),
            ("Better (Small)", "small", "Balance of speed and accuracy"),
            ("Best (Medium)", "medium", "Most accurate, slower processing")
        ]
        
        model_options_frame = ttk.Frame(quality_frame, style="TFrame")
        model_options_frame.pack(fill=tk.X, padx=(10, 0))
        
        for i, (text, value, tooltip) in enumerate(models):
            model_radio_frame = ttk.Frame(model_options_frame, style="TFrame")
            model_radio_frame.pack(anchor=tk.W, pady=3)
            
            radio = ttk.Radiobutton(model_radio_frame, 
                                  text=text, 
                                  variable=self.model_var, 
                                  value=value)
            radio.pack(side=tk.LEFT)
            
            ttk.Label(model_radio_frame, 
                    text=f"({tooltip})", 
                    foreground=self.colors["text_secondary"],
                    background=self.colors["background"],
                    font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=5)
        
        # Chunk size setting with better layout
        chunk_frame = ttk.Frame(settings_frame, style="TFrame")
        chunk_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(chunk_frame, 
                text="Processing Chunk Size:", 
                font=("Segoe UI", 11),
                foreground=self.colors["text_primary"],
                background=self.colors["background"]).pack(anchor=tk.W, pady=(0, 5))
        
        self.chunk_size_var = tk.StringVar(value="30")
        chunk_options_frame = ttk.Frame(chunk_frame, style="TFrame")
        chunk_options_frame.pack(fill=tk.X, padx=(10, 0))
        
        chunk_sizes = [
            ("30s", "30"),
            ("1min", "60"),
            ("2min", "120"),
            ("5min", "300")
        ]
        
        for text, value in chunk_sizes:
            ttk.Radiobutton(chunk_options_frame, 
                          text=text, 
                          variable=self.chunk_size_var, 
                          value=value).pack(side=tk.LEFT, padx=(0, 15), pady=3)
        
        # Add separator
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # Progress section with improved visuals
        progress_frame = ttk.Frame(left_panel, style="TFrame")
        progress_frame.pack(fill=tk.X)
        
        ttk.Label(progress_frame, 
                text="Progress", 
                font=("Segoe UI", 14, "bold"),
                foreground=self.colors["text_primary"],
                background=self.colors["background"]).pack(anchor=tk.W, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var, 
                                          maximum=100, 
                                          mode='determinate',
                                          style="Horizontal.TProgressbar",
                                          length=200)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.step_label = ttk.Label(progress_frame, 
                                   text="Ready to start", 
                                   font=("Segoe UI", 10),
                                   foreground=self.colors["text_secondary"],
                                   background=self.colors["background"],
                                   wraplength=250)
        self.step_label.pack(fill=tk.X)
        
        # Control buttons with improved layout
        button_frame = ttk.Frame(left_panel, style="TFrame")
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.transcribe_button = ttk.Button(button_frame, 
                                          text="Start Transcription", 
                                          command=self.start_processing, 
                                        style="Secondary.TButton")
        self.transcribe_button.pack(fill=tk.X, pady=(0, 10))
        
        buttons_row = ttk.Frame(button_frame, style="TFrame")
        buttons_row.pack(fill=tk.X)
        buttons_row.columnconfigure(0, weight=1)
        buttons_row.columnconfigure(1, weight=1)
        buttons_row.columnconfigure(2, weight=1)
        
        self.cancel_button = ttk.Button(buttons_row, 
                                      text="Stop", 
                                      command=self.cancel_processing, 
                                      style="Secondary.TButton",
                                      state='disabled')
        self.cancel_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=2)
        
        ttk.Button(buttons_row, 
                  text="Clear", 
                  command=self.clear_output, 
                  style="Secondary.TButton").grid(row=0, column=1, sticky=(tk.W, tk.E), padx=2)
        
        ttk.Button(buttons_row, 
                  text="Save", 
                  command=self.save_all, 
                  style="Secondary.TButton").grid(row=0, column=2, sticky=(tk.W, tk.E), padx=2)

        # Right panel with notebook - improved styling
        notebook_frame = ttk.Frame(main_content, style="TFrame")
        notebook_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Custom notebook style
        style.configure("TNotebook", background=self.colors["background"])
        style.configure("TNotebook.Tab", 
                      font=("Segoe UI", 11),
                      padding=[12, 6],
                      background=self.colors["background"])
        
        style.map("TNotebook.Tab",
                background=[("selected", self.colors["background"])],
                foreground=[("selected", self.colors["primary"])])
        
        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(expand=True, fill='both')
        
        # Transcription tab
        transcription_frame = ttk.Frame(notebook, style="TFrame", padding=10)
        notebook.add(transcription_frame, text="Transcription")
        
        self.output_text = ScrolledText(transcription_frame, 
                                      wrap=tk.WORD, 
                                      font=("Segoe UI", 11),
                                      bg=self.colors["background"], 
                                      fg=self.colors["text_primary"],
                                      borderwidth=1,
                                      relief="solid",
                                      padx=10,
                                      pady=10)
        self.output_text.pack(expand=True, fill='both')
        
        # Summary tab
        summary_frame = ttk.Frame(notebook, style="TFrame", padding=10)
        notebook.add(summary_frame, text="Summary")
        
        self.summary_text = ScrolledText(summary_frame, 
                                       wrap=tk.WORD, 
                                       font=("Segoe UI", 11),
                                       bg=self.colors["background"], 
                                       fg=self.colors["text_primary"],
                                       borderwidth=1,
                                       relief="solid",
                                       padx=10,
                                       pady=10)
        self.summary_text.pack(expand=True, fill='both')
          # Status bar with improved styling
        self.status_var = tk.StringVar(value="Ready")
        status_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        self.status_bar = ttk.Label(status_frame, 
                                  textvariable=self.status_var,
                                  font=("Segoe UI", 9),
                                  foreground=self.colors["text_secondary"],
                                  background=self.colors["background"],
                                  padding=(10, 5))
        self.status_bar.pack(side=tk.LEFT)
        
        # Initialize models and state
        self.whisper_model = None
        self.summarizer = None
        self.processing = False
        self.cancelled = False
        
        # Start queue processing
        self.process_queue()        # Bind mousewheel scrolling
        self._bind_mousewheel(self.scrollable_frame)
        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._on_mousewheel))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))
        
    def _configure_canvas(self, event):
        """Configure the canvas to expand content to full width"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _bind_mousewheel(self, widget):
        """Bind mousewheel to a widget and all its children"""
        widget.bind('<MouseWheel>', self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_mousewheel(child)

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
                self.update_ui(status="Loading Whisper model...", progress=0, step="Initializing model...")
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
                        chunk_text = f"[{timestamp}] {result['text']}\n\n"
                        
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
                ("All files", ".")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            self.update_ui(status=f"Selected: {os.path.basename(file_path)}")

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.update_ui(status="Ready", step="Ready to start")

    def save_all(self):
        if not self.output_text.get(1.0, tk.END).strip() and not self.summary_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No content to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
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
    root = ThemedTk(theme="arc")
    root.geometry("1200x800")  # Set initial size
    
    # Make window responsive
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    app = AudioTranscriptionApp(root)
    root.mainloop()
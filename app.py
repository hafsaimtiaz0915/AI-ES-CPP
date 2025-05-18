import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import whisper
import threading
import torch
from transformers import pipeline

class AudioTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription & Summarization Tool")
        self.root.geometry("1000x800")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(main_frame, text="Select Audio/Video File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Transcription options
        options_frame = ttk.LabelFrame(main_frame, text="Transcription Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.model_var = tk.StringVar(value="base")
        ttk.Label(options_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(options_frame, text="Base", variable=self.model_var, value="base").grid(row=0, column=1)
        ttk.Radiobutton(options_frame, text="Small", variable=self.model_var, value="small").grid(row=0, column=2)
        ttk.Radiobutton(options_frame, text="Medium", variable=self.model_var, value="medium").grid(row=0, column=3)
        
        # Progress frame
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.progress_frame.grid_remove()  # Hidden by default

        # Create notebook for multiple tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=4, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Transcription tab
        transcription_frame = ttk.Frame(notebook)
        notebook.add(transcription_frame, text="Transcription")
        
        self.output_text = ScrolledText(transcription_frame, height=20, width=70)
        self.output_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        self.summary_text = ScrolledText(summary_frame, height=20, width=70)
        self.summary_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.transcribe_button = ttk.Button(button_frame, text="Start Transcription", command=self.start_transcription)
        self.transcribe_button.pack(side=tk.LEFT, padx=5)
        self.summarize_button = ttk.Button(button_frame, text="Generate Summary", command=self.generate_summary)
        self.summarize_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save All", command=self.save_all).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Initialize models
        self.whisper_model = None
        self.summarizer = None
        self.is_transcribing = False
        self.is_summarizing = False

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Audio/Video files", "*.mp3 *.mp4 *.wav *.m4a *.wma"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path.set(file_path)
            self.status_var.set(f"Selected file: {os.path.basename(file_path)}")
    
    def load_model(self):
        try:
            model_name = self.model_var.get()
            self.status_var.set(f"Loading {model_name} model...")
            self.whisper_model = whisper.load_model(model_name)
            self.status_var.set("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
            return False
        return True

    def transcribe(self):
        try:
            self.progress_frame.grid()  # Show progress bar
            self.progress_var.set(0)
            
            if not self.whisper_model:
                if not self.load_model():
                    return
            
            file_path = self.file_path.get()
            self.status_var.set("Transcribing... This may take a while")
            self.output_text.insert(tk.END, "Starting transcription...\n")
            
            # Perform transcription
            result = self.whisper_model.transcribe(file_path)
            
            # Update UI with result
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result["text"])
            self.status_var.set("Transcription completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Transcription failed: {str(e)}")
            self.status_var.set("Transcription failed")
        finally:
            self.progress_frame.grid_remove()  # Hide progress bar
            self.is_transcribing = False
            self.transcribe_button.configure(state="normal")
    
    def start_transcription(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        if self.is_transcribing:
            return
        
        self.is_transcribing = True
        self.transcribe_button.configure(state="disabled")
        
        # Start transcription in a separate thread
        threading.Thread(target=self.transcribe, daemon=True).start()
    
    def generate_summary(self):
        if not self.output_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No text to summarize.")
            return
            
        if self.is_summarizing:
            return
            
        self.is_summarizing = True
        self.summarize_button.configure(state="disabled")
        threading.Thread(target=self._generate_summary, daemon=True).start()
    
    def _generate_summary(self):
        try:
            self.status_var.set("Loading summarization model...")
            if not self.summarizer:
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            text = self.output_text.get(1.0, tk.END).strip()
            self.status_var.set("Generating summary...")
            
            # Split text into chunks if it's too long
            max_chunk_length = 1024
            chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []
            
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            final_summary = " ".join(summaries)
            
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, final_summary)
            self.status_var.set("Summary generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Summary generation failed: {str(e)}")
            self.status_var.set("Summary generation failed")
        finally:
            self.is_summarizing = False
            self.summarize_button.configure(state="normal")
    
    def save_all(self):
        if not self.output_text.get(1.0, tk.END).strip() and not self.summary_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No content to save.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("=== TRANSCRIPTION ===\n\n")
                    f.write(self.output_text.get(1.0, tk.END))
                    f.write("\n\n=== SUMMARY ===\n\n")
                    f.write(self.summary_text.get(1.0, tk.END))
                self.status_var.set(f"Content saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save content: {str(e)}")

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.summary_text.delete(1.0, tk.END)
        self.status_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTranscriptionApp(root)
    root.mainloop()

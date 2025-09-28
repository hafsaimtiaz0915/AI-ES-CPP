import os
import sys
import ffmpeg
import whisper
import torch
from transformers import pipeline
import numpy as np
from datetime import timedelta

# Structured NLP imports
try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    from gensim import corpora, models
    from collections import Counter
    import re
    STRUCTURED_NLP_AVAILABLE = True
except ImportError as e:
    print(f"[!] Structured NLP libraries not available: {e}")
    print("[*] Install with: pip install spacy scikit-learn networkx gensim")
    STRUCTURED_NLP_AVAILABLE = False

# Ensure output encoding is UTF-8 (fix for Windows emoji/Unicode errors)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass  # For Python <3.7

def get_video_duration(video_path):
    """Get the duration of the video using ffmpeg"""
    probe = ffmpeg.probe(video_path)
    duration = float(probe['streams'][0]['duration'])
    return duration

def extract_audio_chunk(video_path, start_time, chunk_duration, output_audio):
    """Extract a chunk of audio from the video"""
    print(f"[*] Extracting audio chunk from {start_time}s to {start_time + chunk_duration}s...")
    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, t=chunk_duration)
            .output(output_audio, acodec='pcm_s16le', ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        print(f"Error extracting audio chunk: {e.stderr.decode()}")
        return False

def transcribe_chunk(model, audio_path, start_time):
    """Transcribe a single audio chunk"""
    result = model.transcribe(
        audio_path,
        initial_prompt="This is a meeting transcript. Please maintain proper punctuation and capitalization.",
        fp16=False  # Explicitly disable FP16
    )
    timestamp = str(timedelta(seconds=int(start_time)))
    return f"[{timestamp}] {result['text']}\n"

# ===== STRUCTURED NLP FUNCTIONS =====

def extract_entities(text):
    """Extract named entities using spaCy"""
    if not STRUCTURED_NLP_AVAILABLE:
        return [("NLP libraries not available", "ERROR")]
    
    try:
        # Try to load English model
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("[!] spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        return [("spaCy model not available", "ERROR")]
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities[:20]  # Return top 20 entities

def textrank_summarization(text, num_sentences=3):
    """Extractive summarization using TextRank algorithm"""
    if not STRUCTURED_NLP_AVAILABLE:
        return "TextRank not available - missing dependencies"
    
    # Split text into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 20]
    
    if len(sentences) < num_sentences:
        return '. '.join(sentences)
    
    try:
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Get top sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        return '. '.join([s for _, s in ranked_sentences[:num_sentences]]) + '.'
    except Exception as e:
        print(f"[!] TextRank error: {e}")
        return "TextRank summarization failed"

def lda_topic_modeling(text, num_topics=3):
    """Topic modeling using Latent Dirichlet Allocation"""
    if not STRUCTURED_NLP_AVAILABLE:
        return [(0, "LDA not available - missing dependencies")]
    
    try:
        # Preprocess text
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        texts = []
        
        for sentence in sentences:
            # Simple tokenization and cleaning
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
            if len(words) > 2:
                texts.append(words)
        
        if len(texts) < 3:
            return [(0, "Insufficient text for topic modeling")]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=min(num_topics, len(texts)), 
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        topics = lda_model.print_topics(num_words=5)
        return topics
    except Exception as e:
        print(f"[!] LDA error: {e}")
        return [(0, "Topic modeling failed")]

def structured_summarization_pipeline(text):
    """Complete structured NLP pipeline"""
    print("[*] Running structured NLP analysis...")
    
    # Extract entities
    entities = extract_entities(text)
    
    # Generate extractive summary
    extractive_summary = textrank_summarization(text, num_sentences=5)
    
    # Topic modeling
    topics = lda_topic_modeling(text)
    
    return {
        'extractive_summary': extractive_summary,
        'entities': entities,
        'topics': topics
    }

def process_video_in_chunks(video_path, chunk_duration=300):  # 5 minutes chunks
    print("[*] Starting video processing...")
    
    if not os.path.exists(video_path):
        print(f"[!] File not found: {video_path}")
        return None, None
    
    # Get video duration
    total_duration = get_video_duration(video_path)
    print(f"[*] Total video duration: {total_duration:.2f} seconds")
    
    # Load Whisper model with explicit CPU and FP32 settings
    print("[*] Loading Whisper model...")
    model = whisper.load_model(
        "base",
        device="cpu",
        download_root=None,
        in_memory=True
    )
    # Force FP32
    model = model.float()
    
    # Initialize transcript
    full_transcript = ""
    temp_audio = "temp_chunk.wav"
    
    # Process video in chunks
    start_time = 0
    while start_time < total_duration:
        current_chunk_duration = min(chunk_duration, total_duration - start_time)
        
        print(f"\n[*] Processing chunk: {start_time:.1f}s to {start_time + current_chunk_duration:.1f}s")
        
        # Extract audio chunk
        if extract_audio_chunk(video_path, start_time, current_chunk_duration, temp_audio):
            # Transcribe chunk
            chunk_transcript = transcribe_chunk(model, temp_audio, start_time)
            full_transcript += chunk_transcript
            
            # Clean up temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        
        start_time += chunk_duration
        print(f"[*] Progress: {min(100, (start_time/total_duration)*100):.1f}%")
    
    # Generate summaries using both approaches
    print("\n[*] Generating summaries...")
    
    # Unstructured (existing) summary
    print("[*] Generating abstractive summary (BART)...")
    abstractive_summary = summarize_text(full_transcript)
    
    # Structured analysis
    print("[*] Running structured NLP pipeline...")
    structured_analysis = structured_summarization_pipeline(full_transcript)
    
    return full_transcript, abstractive_summary, structured_analysis

def summarize_text(text, max_words=800):
    print("[*] Summarizing transcript with Transformers...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    words = text.split()
    
    # Process long text in chunks if needed
    if len(words) > max_words:
        summaries = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunk_summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(chunk_summary[0]['summary_text'])
        return " ".join(summaries)
    else:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

def save_output(transcript, abstractive_summary, structured_analysis, filename="transcript_summary.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== FULL TRANSCRIPT (with timestamps) ===\n\n")
        f.write(transcript + "\n\n")
        
        f.write("=== ABSTRACTIVE SUMMARY (BART) ===\n\n")
        f.write(abstractive_summary + "\n\n")
        
        f.write("=== EXTRACTIVE SUMMARY (TextRank) ===\n\n")
        f.write(structured_analysis['extractive_summary'] + "\n\n")
        
        f.write("=== NAMED ENTITIES ===\n\n")
        for entity, label in structured_analysis['entities']:
            f.write(f"{entity} ({label})\n")
        
        f.write("\n=== TOPICS (LDA) ===\n\n")
        for i, topic in enumerate(structured_analysis['topics']):
            f.write(f"Topic {i+1}: {topic[1]}\n")
            
    print(f"[✓] Complete analysis saved to: {filename}")

def summarize_meeting(video_path):
    result = process_video_in_chunks(video_path)
    if result and len(result) == 3:
        transcript, abstractive_summary, structured_analysis = result
        if transcript and abstractive_summary:
            save_output(transcript, abstractive_summary, structured_analysis)
            print("[✓] Video processing complete!")
            print(f"[*] Abstractive summary: {len(abstractive_summary.split())} words")
            print(f"[*] Extractive summary: {len(structured_analysis['extractive_summary'].split())} words")
            print(f"[*] Entities found: {len(structured_analysis['entities'])}")
            print(f"[*] Topics identified: {len(structured_analysis['topics'])}")
        else:
            print("[!] Processing failed.")
    else:
        print("[!] Processing failed - invalid result format.")

# Replace this with your actual video file name
if __name__ == "__main__":
    video_path = "video.mp4"  # 🔁 Change this filename to your real video
    summarize_meeting(video_path)
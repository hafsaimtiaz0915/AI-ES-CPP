# Evaluation Metrics for NLP Project
# Comparative Study of Structured and Unstructured NLP Models

import time
import psutil
import os
from typing import Dict, Any, Tuple, Optional

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("[!] ROUGE not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from sacrebleu import sentence_bleu
    BLEU_AVAILABLE = True
except ImportError:
    print("[!] BLEU not available. Install with: pip install sacrebleu")
    BLEU_AVAILABLE = False

class EvaluationMetrics:
    """
    Comprehensive evaluation system for comparing structured vs unstructured NLP methods
    """
    
    def __init__(self):
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def calculate_rouge_scores(self, generated_summary: str, reference_summary: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores for summary evaluation
        
        Args:
            generated_summary: AI-generated summary
            reference_summary: Human-written reference summary
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        if not ROUGE_AVAILABLE or not self.rouge_scorer:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'error': 'ROUGE not available'
            }
        
        try:
            scores = self.rouge_scorer.score(reference_summary, generated_summary)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'error': str(e)
            }
    
    def calculate_bleu_score(self, generated_text: str, reference_text: str) -> float:
        """
        Calculate BLEU score for text comparison
        
        Args:
            generated_text: AI-generated text
            reference_text: Reference text
            
        Returns:
            BLEU score (0-1)
        """
        if not BLEU_AVAILABLE:
            return 0.0
        
        try:
            return sentence_bleu([reference_text.split()], generated_text.split())
        except Exception as e:
            print(f"[!] BLEU calculation error: {e}")
            return 0.0
    
    def measure_processing_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure function execution time
        
        Args:
            func: Function to measure
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (function_result, execution_time_seconds)
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            return result, end_time - start_time
        except Exception as e:
            end_time = time.time()
            return None, end_time - start_time
    
    def measure_memory_usage(self) -> float:
        """
        Get current memory usage in MB
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            print(f"[!] Memory measurement error: {e}")
            return 0.0
    
    def evaluate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Evaluate basic text quality metrics
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        if not text or not text.strip():
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'readability_score': 0,
                'error': 'Empty text'
            }
        
        # Basic metrics
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Simple readability approximation (Flesch Reading Ease approximation)
        avg_words_per_sentence = avg_sentence_length
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words) if words else 0
        
        # Simplified Flesch score
        readability_score = max(0, min(100, 
            206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        ))
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'readability_score': round(readability_score, 2)
        }
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting approximation"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        prev_char_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_vowel:
                    syllables += 1
                prev_char_vowel = True
            else:
                prev_char_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
            
        return max(1, syllables)
    
    def comparative_evaluation(self, 
                             text: str, 
                             structured_func, 
                             unstructured_func,
                             reference_summary: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare structured vs unstructured methods comprehensively
        
        Args:
            text: Input text to process
            structured_func: Function implementing structured approach
            unstructured_func: Function implementing unstructured approach
            reference_summary: Optional reference summary for ROUGE evaluation
            
        Returns:
            Comprehensive comparison results
        """
        print("[*] Starting comparative evaluation...")
        
        results = {
            'structured': {},
            'unstructured': {},
            'comparison': {}
        }
        
        # Measure structured approach
        print("[*] Evaluating structured approach...")
        memory_before_structured = self.measure_memory_usage()
        structured_result, structured_time = self.measure_processing_time(structured_func, text)
        memory_after_structured = self.measure_memory_usage()
        
        results['structured'] = {
            'result': structured_result,
            'processing_time': round(structured_time, 3),
            'memory_used': round(memory_after_structured - memory_before_structured, 2),
            'quality_metrics': self.evaluate_text_quality(
                structured_result.get('extractive_summary', '') if isinstance(structured_result, dict) else str(structured_result)
            )
        }
        
        # Measure unstructured approach
        print("[*] Evaluating unstructured approach...")
        memory_before_unstructured = self.measure_memory_usage()
        unstructured_result, unstructured_time = self.measure_processing_time(unstructured_func, text)
        memory_after_unstructured = self.measure_memory_usage()
        
        results['unstructured'] = {
            'result': unstructured_result,
            'processing_time': round(unstructured_time, 3),
            'memory_used': round(memory_after_unstructured - memory_before_unstructured, 2),
            'quality_metrics': self.evaluate_text_quality(str(unstructured_result))
        }
        
        # Comparative analysis
        results['comparison'] = {
            'speed_ratio': round(structured_time / unstructured_time if unstructured_time > 0 else 0, 2),
            'memory_ratio': round(
                (memory_after_structured - memory_before_structured) / 
                (memory_after_unstructured - memory_before_unstructured) 
                if (memory_after_unstructured - memory_before_unstructured) > 0 else 0, 2
            ),
            'faster_method': 'structured' if structured_time < unstructured_time else 'unstructured',
            'more_efficient_memory': 'structured' if (memory_after_structured - memory_before_structured) < (memory_after_unstructured - memory_before_unstructured) else 'unstructured'
        }
        
        # ROUGE evaluation if reference available
        if reference_summary and ROUGE_AVAILABLE:
            structured_summary = structured_result.get('extractive_summary', '') if isinstance(structured_result, dict) else str(structured_result)
            unstructured_summary = str(unstructured_result)
            
            results['structured']['rouge_scores'] = self.calculate_rouge_scores(structured_summary, reference_summary)
            results['unstructured']['rouge_scores'] = self.calculate_rouge_scores(unstructured_summary, reference_summary)
            
            # BLEU scores
            if BLEU_AVAILABLE:
                results['structured']['bleu_score'] = self.calculate_bleu_score(structured_summary, reference_summary)
                results['unstructured']['bleu_score'] = self.calculate_bleu_score(unstructured_summary, reference_summary)
        
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                 output_file: str = "evaluation_report.txt") -> None:
        """
        Generate a comprehensive evaluation report
        
        Args:
            evaluation_results: Results from comparative_evaluation
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== COMPARATIVE EVALUATION REPORT ===\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Structured Processing Time: {evaluation_results['structured']['processing_time']}s\n")
            f.write(f"Unstructured Processing Time: {evaluation_results['unstructured']['processing_time']}s\n")
            f.write(f"Speed Ratio (S/U): {evaluation_results['comparison']['speed_ratio']}\n")
            f.write(f"Faster Method: {evaluation_results['comparison']['faster_method']}\n\n")
            
            # Memory usage
            f.write("MEMORY USAGE:\n")
            f.write(f"Structured Memory: {evaluation_results['structured']['memory_used']} MB\n")
            f.write(f"Unstructured Memory: {evaluation_results['unstructured']['memory_used']} MB\n")
            f.write(f"Memory Ratio (S/U): {evaluation_results['comparison']['memory_ratio']}\n")
            f.write(f"More Memory Efficient: {evaluation_results['comparison']['more_efficient_memory']}\n\n")
            
            # Quality metrics
            f.write("TEXT QUALITY METRICS:\n")
            f.write("Structured Approach:\n")
            for metric, value in evaluation_results['structured']['quality_metrics'].items():
                f.write(f"  {metric}: {value}\n")
            
            f.write("Unstructured Approach:\n")
            for metric, value in evaluation_results['unstructured']['quality_metrics'].items():
                f.write(f"  {metric}: {value}\n")
            
            # ROUGE scores if available
            if 'rouge_scores' in evaluation_results['structured']:
                f.write("\nROUGE SCORES:\n")
                f.write("Structured:\n")
                for metric, score in evaluation_results['structured']['rouge_scores'].items():
                    f.write(f"  {metric}: {score:.4f}\n")
                
                f.write("Unstructured:\n")
                for metric, score in evaluation_results['unstructured']['rouge_scores'].items():
                    f.write(f"  {metric}: {score:.4f}\n")
            
            # BLEU scores if available
            if 'bleu_score' in evaluation_results['structured']:
                f.write("\nBLEU SCORES:\n")
                f.write(f"Structured: {evaluation_results['structured']['bleu_score']:.4f}\n")
                f.write(f"Unstructured: {evaluation_results['unstructured']['bleu_score']:.4f}\n")
        
        print(f"[✓] Evaluation report saved to: {output_file}")


# Example usage and testing
if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    
    # Test basic functionality
    sample_text = "This is a sample text for testing. It contains multiple sentences. Each sentence should be evaluated properly."
    quality_metrics = evaluator.evaluate_text_quality(sample_text)
    
    print("Text Quality Metrics:", quality_metrics)
    
    if ROUGE_AVAILABLE:
        rouge_scores = evaluator.calculate_rouge_scores(
            "This is a generated summary.",
            "This is the reference summary."
        )
        print("ROUGE Scores:", rouge_scores)
    
    print("[✓] Evaluation metrics module ready!")
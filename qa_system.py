# Question-Answering System for NLP Project
# Implements both structured and unstructured Q&A approaches

import difflib
from typing import List, Dict, Any, Tuple
from collections import Counter
import re

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[!] Transformers not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("[!] spaCy not available. Install with: pip install spacy")
    SPACY_AVAILABLE = False

class QuestionAnsweringSystem:
    """
    Dual-approach Question-Answering system supporting both structured and unstructured methods
    """
    
    def __init__(self):
        self.qa_pipeline = None
        self.nlp = None
        self.context = ""
        self.sentences = []
        self.entities = []
        
    def set_context(self, text: str) -> None:
        """
        Set the context text for question answering
        
        Args:
            text: Context text (transcript or document)
        """
        self.context = text
        self.sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Extract entities if spaCy is available
        if SPACY_AVAILABLE and self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                doc = self.nlp(text)
                self.entities = [(ent.text.lower(), ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
            except OSError:
                print("[!] spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.entities = []
        
    def load_qa_model(self, model_name: str = "distilbert-base-cased-distilled-squad") -> bool:
        """
        Load the transformer-based question-answering model
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            print("[!] Transformers library not available")
            return False
        
        try:
            print(f"[*] Loading Q&A model: {model_name}...")
            self.qa_pipeline = pipeline("question-answering", model=model_name)
            print("[✓] Q&A model loaded successfully")
            return True
        except Exception as e:
            print(f"[!] Error loading Q&A model: {e}")
            return False
    
    def structured_qa(self, question: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Structured keyword-based question answering
        
        Args:
            question: User's question
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.context:
            return {
                'answer': "No context available for answering questions.",
                'confidence': 0.0,
                'method': 'structured',
                'sources': [],
                'error': 'No context'
            }
        
        # Preprocess question
        question_lower = question.lower()
        question_words = set(re.findall(r'\\b\\w{3,}\\b', question_lower))
        
        # Remove common question words
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'are', 'is', 'was', 'were', 'the', 'and', 'or'}
        question_keywords = question_words - stop_words
        
        if not question_keywords:
            return {
                'answer': "Unable to identify key terms in the question.",
                'confidence': 0.0,
                'method': 'structured',
                'sources': [],
                'error': 'No keywords found'
            }
        
        # Find matching sentences
        sentence_scores = []
        
        for i, sentence in enumerate(self.sentences):
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\\b\\w{3,}\\b', sentence_lower))
            
            # Calculate keyword overlap
            overlap = len(question_keywords & sentence_words)
            
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, question_lower, sentence_lower).ratio()
            
            # Entity matching bonus
            entity_bonus = 0
            for entity_text, entity_label, _, _ in self.entities:
                if any(keyword in entity_text for keyword in question_keywords):
                    if entity_text in sentence_lower:
                        entity_bonus += 0.2
            
            # Combined score
            combined_score = (overlap * 2) + (similarity * 3) + entity_bonus
            
            if combined_score > 0:
                sentence_scores.append((combined_score, sentence, i))
        
        # Sort by score and get top results
        sentence_scores.sort(reverse=True)
        
        if not sentence_scores:
            return {
                'answer': "No relevant information found in the context.",
                'confidence': 0.0,
                'method': 'structured',
                'sources': [],
                'error': 'No matches'
            }
        
        # Combine top sentences for answer
        top_sentences = sentence_scores[:max_results]
        answer_parts = [sentence for _, sentence, _ in top_sentences]
        combined_answer = '. '.join(answer_parts)
        
        # Calculate confidence based on top score
        max_score = top_sentences[0][0]
        confidence = min(1.0, max_score / 10.0)  # Normalize to 0-1
        
        # Create source information
        sources = [
            {
                'sentence_index': idx,
                'score': score,
                'text': sentence[:100] + "..." if len(sentence) > 100 else sentence
            }
            for score, sentence, idx in top_sentences
        ]
        
        return {
            'answer': combined_answer,
            'confidence': round(confidence, 3),
            'method': 'structured',
            'sources': sources,
            'keywords_matched': list(question_keywords)
        }
    
    def unstructured_qa(self, question: str) -> Dict[str, Any]:
        """
        Transformer-based question answering
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not TRANSFORMERS_AVAILABLE or not self.qa_pipeline:
            return {
                'answer': "Transformer-based Q&A not available. Please load the model first.",
                'confidence': 0.0,
                'method': 'unstructured',
                'error': 'Model not available'
            }
        
        if not self.context:
            return {
                'answer': "No context available for answering questions.",
                'confidence': 0.0,
                'method': 'unstructured',
                'error': 'No context'
            }
        
        try:
            # Handle long context by chunking
            max_length = 512  # BERT-style models typically handle ~512 tokens
            
            if len(self.context) <= max_length:
                # Process entire context
                result = self.qa_pipeline(question=question, context=self.context)
                return {
                    'answer': result['answer'],
                    'confidence': round(result['score'], 3),
                    'method': 'unstructured',
                    'model_info': {
                        'start_char': result['start'],
                        'end_char': result['end'],
                        'context_length': len(self.context)
                    }
                }
            else:
                # Chunk the context and find best answer
                chunk_size = max_length - len(question) - 50  # Leave room for question and tokens
                chunks = [self.context[i:i+chunk_size] for i in range(0, len(self.context), chunk_size)]
                
                best_answer = ""
                best_score = 0.0
                best_chunk_idx = 0
                
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                        
                    try:
                        result = self.qa_pipeline(question=question, context=chunk)
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_answer = result['answer']
                            best_chunk_idx = i
                    except Exception as chunk_error:
                        print(f"[!] Error processing chunk {i}: {chunk_error}")
                        continue
                
                return {
                    'answer': best_answer if best_answer else "Unable to find answer in the context.",
                    'confidence': round(best_score, 3),
                    'method': 'unstructured',
                    'model_info': {
                        'total_chunks': len(chunks),
                        'best_chunk': best_chunk_idx,
                        'context_length': len(self.context)
                    }
                }
                
        except Exception as e:
            return {
                'answer': f"Error during question answering: {str(e)}",
                'confidence': 0.0,
                'method': 'unstructured',
                'error': str(e)
            }
    
    def answer_question(self, question: str, method: str = "both") -> Dict[str, Any]:
        """
        Answer question using specified method(s)
        
        Args:
            question: User's question
            method: 'structured', 'unstructured', or 'both'
            
        Returns:
            Dictionary with results from requested method(s)
        """
        results = {}
        
        if method in ["structured", "both"]:
            print(f"[*] Processing question with structured method...")
            results['structured'] = self.structured_qa(question)
        
        if method in ["unstructured", "both"]:
            print(f"[*] Processing question with unstructured method...")
            if not self.qa_pipeline:
                print("[*] Loading Q&A model...")
                self.load_qa_model()
            results['unstructured'] = self.unstructured_qa(question)
        
        # Add comparison if both methods used
        if method == "both" and 'structured' in results and 'unstructured' in results:
            results['comparison'] = {
                'structured_confidence': results['structured']['confidence'],
                'unstructured_confidence': results['unstructured']['confidence'],
                'recommended_method': 'unstructured' if results['unstructured']['confidence'] > results['structured']['confidence'] else 'structured'
            }
        
        return results
    
    def batch_qa(self, questions: List[str], method: str = "both") -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            method: 'structured', 'unstructured', or 'both'
            
        Returns:
            List of results for each question
        """
        results = []
        
        for i, question in enumerate(questions):
            print(f"[*] Processing question {i+1}/{len(questions)}: {question[:50]}...")
            result = self.answer_question(question, method)
            result['question'] = question
            result['question_index'] = i
            results.append(result)
        
        return results
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the current context
        
        Returns:
            Dictionary with context statistics
        """
        if not self.context:
            return {'error': 'No context set'}
        
        words = self.context.split()
        
        return {
            'total_characters': len(self.context),
            'total_words': len(words),
            'total_sentences': len(self.sentences),
            'total_entities': len(self.entities),
            'avg_sentence_length': round(len(words) / len(self.sentences), 2) if self.sentences else 0,
            'top_entities': [entity[0] for entity in self.entities[:10]] if self.entities else []
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize Q&A system
    qa_system = QuestionAnsweringSystem()
    
    # Set sample context
    sample_context = """
    This is a meeting about project planning. John discussed the budget requirements.
    The project deadline is December 15th. Sarah mentioned that we need additional resources.
    The total budget is $50,000. Marketing team will handle the promotional activities.
    Development phase should start next week. Testing will begin in November.
    """
    
    qa_system.set_context(sample_context)
    
    # Test questions
    test_questions = [
        "What is the project deadline?",
        "Who discussed the budget?",
        "How much is the total budget?",
        "When does testing begin?"
    ]
    
    print("=== Q&A System Testing ===")
    print(f"Context: {sample_context[:100]}...")
    print(f"Context summary: {qa_system.get_context_summary()}")
    
    for question in test_questions:
        print(f"\\nQuestion: {question}")
        
        # Test structured approach
        structured_result = qa_system.structured_qa(question)
        print(f"Structured answer: {structured_result['answer']} (confidence: {structured_result['confidence']})")
        
        # Test unstructured approach (if available)
        if TRANSFORMERS_AVAILABLE:
            qa_system.load_qa_model()
            unstructured_result = qa_system.unstructured_qa(question)
            print(f"Unstructured answer: {unstructured_result['answer']} (confidence: {unstructured_result['confidence']})")
    
    print("[✓] Q&A System ready!")
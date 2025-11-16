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
    
    def structured_qa(self, question: str) -> Dict:
        """Answer using structured NLP (keyword matching + sentence extraction)"""
        try:
            # Extract keywords from question
            question_lower = question.lower()
            keywords = re.findall(r'\b\w{2,}\b', question_lower)  # Min 2 chars to catch "on", "do", etc.
            
            # Remove common stop words
            stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why', 'is', 'are', 
                         'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'from', 'as', 'be', 'been', 'has', 'had'}
            
            keywords = [k for k in keywords if k not in stop_words]
            
            if not keywords:
                keywords = question_lower.split()  # Fallback to all words
            
            if not keywords:
                return {
                    'answer': "Unable to identify key terms in the question.",
                    'confidence': 0.0,
                    'method': 'structured'
                }
            
            print(f"[*] Extracted keywords: {keywords}")
            
            # Score sentences based on keyword matches with TF-IDF weighting
            sentence_scores = []
            keyword_idf = {}  # Inverse document frequency for keywords
            
            # Calculate IDF for each keyword (rarity bonus)
            for keyword in keywords:
                doc_freq = sum(1 for sent in self.sentences if keyword in sent.lower())
                keyword_idf[keyword] = 1.0 + (len(self.sentences) / max(1, doc_freq))
            
            for sentence in self.sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                sentence_lower = sentence.lower()
                sentence_words = sentence_lower.split()
                
                # Count exact keyword matches with IDF weighting
                keyword_matches = 0
                weighted_matches = 0.0
                
                for keyword in keywords:
                    if keyword in sentence_lower:
                        keyword_matches += 1
                        # TF: term frequency in sentence
                        tf = sentence_words.count(keyword) / len(sentence_words)
                        # Weighted by IDF
                        weighted_matches += tf * keyword_idf[keyword]
                
                # Calculate match ratio (coverage of question keywords)
                match_ratio = keyword_matches / len(keywords) if keywords else 0
                
                # Position bonus: prefer sentences earlier in context (more relevant)
                position_bonus = 0.1 if self.sentences.index(sentence) < len(self.sentences) * 0.3 else 0
                
                # Combined score with multiple factors
                # - keyword_matches * 3: raw match count (most important)
                # - match_ratio * 6: coverage of all question keywords
                # - weighted_matches * 2: TF-IDF weighted importance
                # - position_bonus: slight preference for earlier sentences
                score = (keyword_matches * 3) + (match_ratio * 6) + (weighted_matches * 2) + position_bonus
                
                if score > 0:
                    sentence_scores.append((score, sentence, keyword_matches, match_ratio))
            
            if not sentence_scores:
                return {
                    'answer': "No relevant information found in the context.",
                    'confidence': 0.0,
                    'method': 'structured'
                }
            
            # Sort by score and get top 2-3 sentences for complete answer
            sentence_scores.sort(reverse=True)
            top_sentences = sentence_scores[:3]
            
            # Combine sentences for complete answer
            answer_parts = [sent for _, sent, _, _ in top_sentences]
            combined_answer = ' '.join(answer_parts)
            
            # Advanced confidence calculation
            max_score = top_sentences[0][0]
            top_match_ratio = top_sentences[0][3]
            top_keyword_count = top_sentences[0][2]
            
            # Multiple factors influence confidence:
            # 1. Base score (primary indicator)
            base_confidence = min(0.85, 0.30 + (max_score / 15.0))
            
            # 2. Keyword coverage bonus (if all keywords matched)
            coverage_bonus = 0.15 if top_match_ratio >= 0.8 else 0.10 if top_match_ratio >= 0.6 else 0.05
            
            # 3. Multiple sentence penalty (combined answers less certain)
            multi_sentence_penalty = 0.05 if len(top_sentences) > 1 else 0
            
            # 4. Strong match bonus (3+ keywords matched)
            strong_match_bonus = 0.10 if top_keyword_count >= 3 else 0.05 if top_keyword_count >= 2 else 0
            
            # Final confidence with all factors
            confidence = base_confidence + coverage_bonus + strong_match_bonus - multi_sentence_penalty
            confidence = min(0.95, max(0.20, confidence))  # Clamp between 20-95%
            
            print(f"[*] Structured Q&A - Score: {max_score:.2f}, Keywords: {top_keyword_count}/{len(keywords)}, "
                  f"Coverage: {top_match_ratio:.0%}, Confidence: {confidence:.1%}")
            
            return {
                'answer': combined_answer,
                'confidence': round(confidence, 3),
                'method': 'structured',
                'keywords_found': keywords
            }
            
        except Exception as e:
            print(f"[!] Error in structured_qa: {e}")
            return {
                'answer': f"Error in structured analysis: {str(e)}",
                'confidence': 0.0,
                'method': 'structured'
            }
    
    def unstructured_qa(self, question: str) -> Dict[str, Any]:
        """
        Transformer-based question answering
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                'answer': "Transformer-based Q&A not available. Transformers library not installed.",
                'confidence': 0.0,
                'method': 'unstructured',
                'error': 'Transformers not available'
            }
        
        # Try to load model if not already loaded
        if not self.qa_pipeline:
            try:
                print("[*] Auto-loading Q&A model...")
                if not self.load_qa_model():
                    return {
                        'answer': "Failed to load Q&A model. Using structured method instead.",
                        'confidence': 0.0,
                        'method': 'unstructured',
                        'error': 'Model loading failed'
                    }
            except Exception as e:
                return {
                    'answer': f"Error loading Q&A model: {str(e)}",
                    'confidence': 0.0,
                    'method': 'unstructured',
                    'error': str(e)
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
                
                # Validate answer relevance
                answer_confidence = self._validate_answer_relevance(question, result['answer'], result['score'])
                
                return {
                    'answer': result['answer'],
                    'confidence': round(answer_confidence, 3),
                    'method': 'unstructured',
                    'model_info': {
                        'start_char': result['start'],
                        'end_char': result['end'],
                        'context_length': len(self.context),
                        'original_confidence': round(result['score'], 3)
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
                
                # Validate answer relevance
                validated_confidence = self._validate_answer_relevance(question, best_answer, best_score)
                
                return {
                    'answer': best_answer if best_answer else "Unable to find answer in the context.",
                    'confidence': round(validated_confidence, 3),
                    'method': 'unstructured',
                    'model_info': {
                        'total_chunks': len(chunks),
                        'best_chunk': best_chunk_idx,
                        'context_length': len(self.context),
                        'original_confidence': round(best_score, 3)
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
    
    def _validate_answer_relevance(self, question: str, answer: str, original_confidence: float) -> float:
        """
        Validate if answer type matches question type and adjust confidence accordingly.
        This prevents high confidence for wrong answer types (e.g., month when day is asked).
        """
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Define question type patterns
        day_questions = ['which day', 'what day', 'on which day', 'on what day', 'day of']
        time_questions = ['what time', 'when', 'which time', 'at what time']
        person_questions = ['who', 'which person', 'what person', 'who is', 'who are']
        location_questions = ['where', 'which place', 'what place', 'in which', 'at which']
        number_questions = ['how many', 'how much', 'what number', 'number of']
        reason_questions = ['why', 'reason', 'because']
        
        # Define answer type patterns
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'weekday', 'weekend']
        months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 
                 'august', 'september', 'october', 'november', 'december']
        
        # Check answer length (too short or too long is suspicious)
        answer_words = answer.split()
        if len(answer.strip()) < 2:
            print(f"[!] Answer too short: '{answer}' - Reducing confidence by 60%")
            return original_confidence * 0.4
        
        if len(answer_words) > 25:
            print(f"[!] Answer too long ({len(answer_words)} words) - Reducing confidence by 30%")
            return original_confidence * 0.7
        
        penalty = 0.0
        
        # TYPE MISMATCH DETECTION
        
        # 1. Day question but got month answer
        if any(dq in question_lower for dq in day_questions):
            has_day = any(day in answer_lower for day in days)
            has_month = any(month in answer_lower for month in months)
            
            if has_month and not has_day:
                penalty = 0.55  # Heavy penalty: 55% reduction
                print(f"[!] TYPE MISMATCH: Day question but got month answer '{answer}' - Reducing confidence by 55%")
            elif not has_day and not has_month:
                # Answer doesn't contain day or month - probably wrong
                penalty = 0.35
                print(f"[!] Day question but answer '{answer}' contains no day - Reducing confidence by 35%")
        
        # 2. Person question but got non-person answer
        elif any(pq in question_lower for pq in person_questions):
            # Person names typically start with capital letter and are short (1-3 words)
            if not any(word[0].isupper() for word in answer_words if len(word) > 1):
                penalty = 0.30
                print(f"[!] Person question but answer '{answer}' has no capitalized names - Reducing confidence by 30%")
            elif len(answer_words) > 5:
                penalty = 0.20
                print(f"[!] Person question but answer is too long - Reducing confidence by 20%")
        
        # 3. Location question
        elif any(lq in question_lower for lq in location_questions):
            # Locations should be proper nouns (capitalized)
            if not any(word[0].isupper() for word in answer_words if len(word) > 1):
                penalty = 0.25
                print(f"[!] Location question but answer '{answer}' has no capitalized places - Reducing confidence by 25%")
        
        # 4. Number question but got text answer
        elif any(nq in question_lower for nq in number_questions):
            # Should contain digits or number words
            number_words = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 
                          'dozen', 'hundred', 'thousand', 'million', 'zero', 'several', 'many', 'few'}
            has_number = any(char.isdigit() for char in answer) or any(nw in answer_lower for nw in number_words)
            
            if not has_number:
                penalty = 0.40
                print(f"[!] Number question but answer '{answer}' contains no numbers - Reducing confidence by 40%")
        
        # 5. Time/When question
        elif any(tq in question_lower for tq in time_questions):
            time_indicators = ['am', 'pm', 'morning', 'afternoon', 'evening', 'night', 'o\'clock', 
                             'midnight', 'noon', ':'] + days + months
            has_time = any(ti in answer_lower for ti in time_indicators) or any(char.isdigit() for char in answer)
            
            if not has_time:
                penalty = 0.30
                print(f"[!] Time question but answer '{answer}' has no time indicators - Reducing confidence by 30%")
        
        # 6. Keyword overlap check (answer should relate to question)
        question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
        answer_words_set = set(re.findall(r'\b\w{3,}\b', answer_lower))
        
        # Remove stop words for better matching
        stop_words = {'what', 'when', 'where', 'which', 'who', 'how', 'why', 'the', 'is', 'are', 'was', 'were'}
        question_words -= stop_words
        answer_words_set -= stop_words
        
        overlap = len(question_words & answer_words_set)
        
        # If no keyword overlap, answer might be off-topic
        if len(question_words) > 0 and overlap == 0:
            penalty = max(penalty, 0.20)
            print(f"[!] No keyword overlap between question and answer - Reducing confidence by 20%")
        
        # Apply penalty
        adjusted_confidence = max(0.05, original_confidence - penalty)
        
        if penalty > 0:
            print(f"[*] Confidence adjusted: {original_confidence:.1%} → {adjusted_confidence:.1%} (penalty: {penalty:.1%})")
        
        return adjusted_confidence


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
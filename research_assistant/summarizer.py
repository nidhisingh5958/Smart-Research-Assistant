"""
Text summarization module for research papers.
"""

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import numpy as np
from transformers import pipeline

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Summarizer:
    """
    Class for summarizing research papers using extractive and abstractive methods.
    """
    
    def __init__(self, use_transformers=True, transformer_model="facebook/bart-large-cnn"):
        """
        Initialize the summarizer.
        
        Args:
            use_transformers (bool): Whether to use transformer models
            transformer_model (str): Transformer model to use for abstractive summarization
        """
        self.stop_words = set(stopwords.words('english'))
        self.use_transformers = use_transformers
        
        # Initialize transformer model if enabled
        if use_transformers:
            try:
                self.summarizer_model = pipeline("summarization", model=transformer_model)
            except Exception as e:
                print(f"Warning: Failed to load transformer model: {str(e)}")
                print("Falling back to extractive summarization only")
                self.use_transformers = False
    
    def extractive_summarize(self, text, ratio=0.3):
        """
        Generate an extractive summary by selecting the most important sentences.
        
        Args:
            text (str): Text to summarize
            ratio (float): Proportion of sentences to include in the summary
            
        Returns:
            str: Extractive summary
        """
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        
        # Skip summarization for very short texts
        if len(sentences) <= 3:
            return text
            
        # Tokenize words and remove stopwords
        words = [word.lower() for word in nltk.word_tokenize(text) 
                if word.isalnum() and word.lower() not in self.stop_words]
        
        # Calculate word frequencies
        freq_dist = FreqDist(words)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = [word.lower() for word in nltk.word_tokenize(sentence) 
                                if word.isalnum()]
            
            score = sum(freq_dist[word] for word in words_in_sentence 
                        if word not in self.stop_words)
            
            # Normalize by sentence length to avoid bias towards longer sentences
            sentence_scores[i] = score / max(len(words_in_sentence), 1)
        
        # Select top sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        top_indices = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Combine selected sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def abstractive_summarize(self, text, max_length=150, min_length=50):
        """
        Generate an abstractive summary using transformer models.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: Abstractive summary
        """
        if not self.use_transformers:
            raise ValueError("Transformer models not available for abstractive summarization")
            
        # Check if text is too long for the model
        # Most models have a limit around 1024 tokens
        tokenized_text = nltk.word_tokenize(text)
        if len(tokenized_text) > 1000:
            # Split into chunks and summarize each chunk
            chunks = self._split_into_chunks(text, 900)
            chunk_summaries = []
            
            for chunk in chunks:
                try:
                    summary = self.summarizer_model(chunk, 
                                                   max_length=max_length // len(chunks),
                                                   min_length=min_length // len(chunks),
                                                   do_sample=False)[0]['summary_text']
                    chunk_summaries.append(summary)
                except Exception as e:
                    print(f"Warning: Error summarizing chunk: {str(e)}")
                    # Fall back to extractive summarization for this chunk
                    chunk_summaries.append(self.extractive_summarize(chunk, ratio=0.3))
            
            return ' '.join(chunk_summaries)
        else:
            try:
                return self.summarizer_model(text, 
                                           max_length=max_length,
                                           min_length=min_length,
                                           do_sample=False)[0]['summary_text']
            except Exception as e:
                print(f"Warning: Error in abstractive summarization: {str(e)}")
                # Fall back to extractive summarization
                return self.extractive_summarize(text, ratio=0.3)
    
    def summarize(self, text, method="hybrid", **kwargs):
        """
        Summarize text using the specified method.
        
        Args:
            text (str): Text to summarize
            method (str): Summarization method ('extractive', 'abstractive', or 'hybrid')
            **kwargs: Additional parameters for the summarization methods
            
        Returns:
            str: Summary
        """
        if method == "extractive":
            ratio = kwargs.get('ratio', 0.3)
            return self.extractive_summarize(text, ratio=ratio)
        elif method == "abstractive":
            if not self.use_transformers:
                print("Warning: Transformer models not available. Using extractive summarization.")
                return self.extractive_summarize(text, ratio=kwargs.get('ratio', 0.3))
            
            max_length = kwargs.get('max_length', 150)
            min_length = kwargs.get('min_length', 50)
            return self.abstractive_summarize(text, max_length=max_length, min_length=min_length)
        elif method == "hybrid":
            # First generate extractive summary to reduce text size
            extractive_ratio = kwargs.get('extractive_ratio', 0.5)
            extractive_summary = self.extractive_summarize(text, ratio=extractive_ratio)
            
            # Then apply abstractive summarization if available
            if self.use_transformers:
                max_length = kwargs.get('max_length', 150)
                min_length = kwargs.get('min_length', 50)
                return self.abstractive_summarize(extractive_summary, 
                                                max_length=max_length,
                                                min_length=min_length)
            else:
                # If transformers not available, return extractive summary with stricter ratio
                return self.extractive_summarize(text, ratio=kwargs.get('ratio', 0.3))
        else:
            raise ValueError(f"Unknown summarization method: {method}")
    
    def _split_into_chunks(self, text, chunk_size):
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text (str): Text to split
            chunk_size (int): Approximate size of each chunk in tokens
            
        Returns:
            list: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(nltk.word_tokenize(sentence))
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def extract_key_points(self, text, num_points=5):
        """
        Extract key points from text.
        
        Args:
            text (str): Text to process
            num_points (int): Number of key points to extract
            
        Returns:
            list: List of key points
        """
        # Use extractive summarization to get important sentences
        sentences = sent_tokenize(text)
        
        # Skip for very short texts
        if len(sentences) <= num_points:
            return sentences
            
        # Tokenize words and remove stopwords
        words = [word.lower() for word in nltk.word_tokenize(text) 
                if word.isalnum() and word.lower() not in self.stop_words]
        
        # Calculate word frequencies
        freq_dist = FreqDist(words)
        
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = [word.lower() for word in nltk.word_tokenize(sentence) 
                                if word.isalnum()]
            
            score = sum(freq_dist[word] for word in words_in_sentence 
                        if word not in self.stop_words)
            
            # Normalize by sentence length to avoid bias towards longer sentences
            sentence_scores[i] = score / max(len(words_in_sentence), 1)
        
        # Select top sentences
        top_indices = nlargest(num_points, sentence_scores, key=sentence_scores.get)
        
        # Sort indices to preserve original order
        top_indices.sort()
        
        # Return selected sentences as key points
        return [sentences[i] for i in top_indices]
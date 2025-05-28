"""
Question answering system for research papers.
"""

import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download required NLTK resources
nltk.download('punkt', quiet=True)

class QuestionAnsweringSystem:
    """
    Class for answering questions about research papers.
    """
    
    def __init__(self, use_transformer=True, model_name="deepset/roberta-base-squad2"):
        """
        Initialize the QA system.
        
        Args:
            use_transformer (bool): Whether to use transformer-based QA
            model_name (str): Model name for transformer-based QA
        """
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize transformer-based QA if enabled
        if use_transformer:
            try:
                self.qa_pipeline = pipeline("question-answering", model=model_name)
            except Exception as e:
                print(f"Warning: Failed to load QA model: {str(e)}")
                print("Falling back to simpler retrieval-based QA")
                self.use_transformer = False
        
        # Document store
        self.documents = {}
    
    def add_document(self, doc_id, doc_content, doc_metadata=None):
        """
        Add document to the QA system.
        
        Args:
            doc_id (str): Document identifier
            doc_content (str): Document content
            doc_metadata (dict, optional): Document metadata
        """
        self.documents[doc_id] = {
            'id': doc_id,
            'content': doc_content,
            'metadata': doc_metadata or {},
            'sentences': sent_tokenize(doc_content),
            'embedding': self.sentence_transformer.encode(doc_content)
        }
    
    def add_documents_from_processor(self, processed_docs):
        """
        Add documents from document processor.
        
        Args:
            processed_docs (dict): Dictionary of processed documents
        """
        for doc_id, doc_data in processed_docs.items():
            self.add_document(
                doc_id,
                doc_data['full_text'],
                doc_data['metadata']
            )
    
    def _retrieve_relevant_passages(self, query, top_k=3, passage_length=3):
        """
        Retrieve relevant passages from documents based on query similarity.
        
        Args:
            query (str): Query string
            top_k (int): Number of passages to retrieve
            passage_length (int): Number of consecutive sentences in each passage
            
        Returns:
            list: List of relevant passages with metadata
        """
        if not self.documents:
            return []
        
        # Encode query
        query_embedding = self.sentence_transformer.encode(query)
        
        # Calculate document similarities
        doc_similarities = {}
        for doc_id, doc_data in self.documents.items():
            similarity = cosine_similarity(
                [query_embedding],
                [doc_data['embedding']]
            )[0][0]
            doc_similarities[doc_id] = similarity
        
        # Sort documents by similarity
        sorted_docs = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Extract passages from top documents
        passages = []
        
        for doc_id, _ in sorted_docs[:top_k]:
            doc_data = self.documents[doc_id]
            sentences = doc_data['sentences']
            
            # Skip documents with too few sentences
            if len(sentences) < passage_length:
                passages.append({
                    'text': ' '.join(sentences),
                    'doc_id': doc_id,
                    'metadata': doc_data['metadata']
                })
                continue
            
            # Create overlapping passages
            for i in range(0, len(sentences) - passage_length + 1):
                passage_text = ' '.join(sentences[i:i + passage_length])
                passages.append({
                    'text': passage_text,
                    'doc_id': doc_id,
                    'metadata': doc_data['metadata'],
                    'position': i
                })
        
        # Sort passages by similarity to query
        passage_embeddings = [self.sentence_transformer.encode(p['text']) for p in passages]
        passage_similarities = cosine_similarity([query_embedding], passage_embeddings)[0]
        
        for i, similarity in enumerate(passage_similarities):
            passages[i]['similarity'] = similarity
        
        passages.sort(key=lambda x: x['similarity'], reverse=True)
        
        return passages[:top_k]
    
    def answer_question(self, question, top_k=3, confidence_threshold=0.1):
        """
        Answer a question using the document collection.
        
        Args:
            question (str): Question to answer
            top_k (int): Number of top passages to consider
            confidence_threshold (float): Minimum confidence threshold for answers
            
        Returns:
            dict: Answer information
        """
        if not self.documents:
            return {
                'answer': "No documents available to answer the question.",
                'confidence': 0.0,
                'source': None
            }
        
        # Retrieve relevant passages
        relevant_passages = self._retrieve_relevant_passages(question, top_k=top_k)
        
        if not relevant_passages:
            return {
                'answer': "Could not find relevant information to answer the question.",
                'confidence': 0.0,
                'source': None
            }
        
        if self.use_transformer:
            # Use transformer-based QA model for each passage
            answers = []
            
            for passage in relevant_passages:
                try:
                    result = self.qa_pipeline(
                        question=question,
                        context=passage['text']
                    )
                    
                    # Store result with source information
                    answers.append({
                        'answer': result['answer'],
                        'confidence': result['score'],
                        'source': {
                            'doc_id': passage['doc_id'],
                            'metadata': passage['metadata'],
                            'context': passage['text'],
                            'similarity': passage.get('similarity', 0.0)
                        }
                    })
                except Exception as e:
                    print(f"Error processing passage: {str(e)}")
                    continue
            
            # Sort answers by confidence
            answers.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return best answer above threshold
            for answer in answers:
                if answer['confidence'] >= confidence_threshold:
                    return answer
            
            # If no answer passes threshold, return the best one with low confidence flag
            if answers:
                answers[0]['low_confidence'] = True
                return answers[0]
            
            # Fall back to retrieval-based approach if no answers found
            pass
        
        # Simple retrieval-based QA (fallback or if transformer-based QA is disabled)
        best_passage = relevant_passages[0]
        confidence = best_passage.get('similarity', 0.5)  # Use similarity as confidence
        
        return {
            'answer': best_passage['text'],
            'confidence': confidence,
            'source': {
                'doc_id': best_passage['doc_id'],
                'metadata': best_passage['metadata'],
                'context': best_passage['text'],
                'similarity': best_passage.get('similarity', 0.0)
            },
            'retrieval_based': True  # Flag indicating this is a retrieval-based answer
        }

    def get_document_metadata(self, doc_id):
        """
        Get metadata for a specific document.
        
        Args:
            doc_id (str): Document identifier
            
        Returns:
            dict: Document metadata or None if not found
        """
        if doc_id in self.documents:
            return self.documents[doc_id].get('metadata', {})
        return None
    
    def search_documents(self, query, top_k=3):
        """
        Search for documents relevant to a query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: Relevant documents with similarity scores
        """
        if not self.documents:
            return []
        
        # Encode query
        query_embedding = self.sentence_transformer.encode(query)
        
        # Calculate document similarities
        results = []
        for doc_id, doc_data in self.documents.items():
            similarity = cosine_similarity(
                [query_embedding],
                [doc_data['embedding']]
            )[0][0]
            
            results.append({
                'doc_id': doc_id,
                'similarity': similarity,
                'metadata': doc_data['metadata'],
                'preview': doc_data['content'][:200] + '...'  # Short preview
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
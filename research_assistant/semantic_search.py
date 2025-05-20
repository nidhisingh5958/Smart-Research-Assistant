"""
Search module for performing semantic search across research papers.
"""

import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    """
    Class for performing semantic search on research documents using embeddings.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
            
        self.document_embeddings = {}
        self.document_texts = {}
        self.document_metadata = {}
    
    def add_document(self, doc_id, doc_content, doc_metadata=None):
        """
        Add a document to the search index.
        
        Args:
            doc_id (str): Unique identifier for the document
            doc_content (str): Text content of the document
            doc_metadata (dict, optional): Metadata for the document
        """
        # Create embedding for the document
        embedding = self.model.encode(doc_content)
        
        # Store document information
        self.document_embeddings[doc_id] = embedding
        self.document_texts[doc_id] = doc_content
        self.document_metadata[doc_id] = doc_metadata or {}
    
    def add_documents_from_processor(self, processed_docs):
        """
        Add multiple documents from the document processor.
        
        Args:
            processed_docs (dict): Dictionary of processed documents from DocumentProcessor
        """
        for doc_id, doc_data in processed_docs.items():
            self.add_document(
                doc_id,
                doc_data["full_text"],
                doc_data["metadata"]
            )
    
    def search(self, query, top_k=5):
        """
        Search for documents matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of tuples containing (doc_id, similarity_score, metadata)
        """
        if not self.document_embeddings:
            return []
            
        # Create embedding for the query
        query_embedding = self.model.encode(query)
        
        results = []
        
        # Calculate similarity scores for all documents
        for doc_id, doc_embedding in self.document_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            
            # Add to results
            results.append((doc_id, similarity, self.document_metadata.get(doc_id, {})))
        
        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return results[:top_k]
    
    def search_within_documents(self, query, top_k=5, chunk_size=500, overlap=100):
        """
        Search within documents by breaking them into chunks.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            
        Returns:
            list: List of tuples containing (doc_id, text_chunk, similarity_score, metadata)
        """
        if not self.document_texts:
            return []
            
        # Create embedding for the query
        query_embedding = self.model.encode(query)
        
        chunks = []
        
        # Break documents into chunks
        for doc_id, text in self.document_texts.items():
            # Skip short documents
            if len(text) < chunk_size:
                chunks.append((doc_id, text, 0, len(text)))
                continue
                
            # Create overlapping chunks
            for i in range(0, len(text) - overlap, chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                chunks.append((doc_id, chunk_text, i, i + len(chunk_text)))
        
        results = []
        
        # Calculate similarity for each chunk
        for doc_id, chunk_text, start_pos, end_pos in chunks:
            # Create embedding for the chunk
            chunk_embedding = self.model.encode(chunk_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embedding.reshape(1, -1)
            )[0][0]
            
            # Add to results
            results.append((
                doc_id,
                chunk_text,
                similarity,
                {
                    "start_position": start_pos,
                    "end_position": end_pos,
                    **self.document_metadata.get(doc_id, {})
                }
            ))
        
        # Sort results by similarity score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k results
        return results[:top_k]
    
    def save_index(self, file_path):
        """
        Save the search index to a file.
        
        Args:
            file_path (str): Path to save the index
        """
        index_data = {
            "embeddings": self.document_embeddings,
            "texts": self.document_texts,
            "metadata": self.document_metadata
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, file_path):
        """
        Load a search index from a file.
        
        Args:
            file_path (str): Path to the index file
        """
        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
                
            self.document_embeddings = index_data["embeddings"]
            self.document_texts = index_data["texts"]
            self.document_metadata = index_data["metadata"]
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")
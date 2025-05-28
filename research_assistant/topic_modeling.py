"""
Topic modeling module for research papers.
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TopicModeling:
    """
    Class for identifying research topics and trends using topic modeling techniques.
    """
    
    def __init__(self):
        """Initialize topic modeling."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.documents = []
        
    def preprocess_text(self, text):
        """
        Preprocess text for topic modeling.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: List of preprocessed tokens
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords, short words, and lemmatize
        preprocessed = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return preprocessed
    
    def add_document(self, doc_id, text, metadata=None):
        """
        Add a document to the corpus.
        
        Args:
            doc_id (str): Document identifier
            text (str): Document text
            metadata (dict, optional): Document metadata
        """
        preprocessed_text = self.preprocess_text(text)
        self.documents.append({
            'id': doc_id,
            'tokens': preprocessed_text,
            'metadata': metadata or {}
        })
        
    def build_model(self, num_topics=10, passes=10):
        """
        Build the topic model.
        
        Args:
            num_topics (int): Number of topics to extract
            passes (int): Number of passes through the corpus during training
            
        Returns:
            object: Trained LDA model
        """
        if not self.documents:
            raise ValueError("No documents added to the model")
            
        # Create dictionary
        tokens_list = [doc['tokens'] for doc in self.documents]
        self.dictionary = corpora.Dictionary(tokens_list)
        
        # Filter out extreme values
        self.dictionary.filter_extremes(no_below=2, no_above=0.9)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tokens_list]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        
        return self.lda_model
    
    def get_topics(self, num_words=10):
        """
        Get the main topics from the model.
        
        Args:
            num_words (int): Number of words to include for each topic
            
        Returns:
            list: List of topics with their top words
        """
        if not self.lda_model:
            raise ValueError("Model not built yet")
            
        topics = []
        for topic_id in range(self.lda_model.num_topics):
            topic_words = self.lda_model.show_topic(topic_id, num_words)
            topics.append({
                'id': topic_id,
                'words': topic_words,
                'label': f"Topic {topic_id}"
            })
            
        return topics
    
    def get_document_topics(self, doc_id=None, threshold=0.1):
        """
        Get topic distribution for a specific document or all documents.
        
        Args:
            doc_id (str, optional): Document identifier. If None, returns for all documents.
            threshold (float): Minimum probability threshold for topics
            
        Returns:
            dict or list: Topic distributions for the requested document(s)
        """
        if not self.lda_model:
            raise ValueError("Model not built yet")
            
        if doc_id is not None:
            # Find the document
            doc_index = next((i for i, doc in enumerate(self.documents) if doc['id'] == doc_id), None)
            if doc_index is None:
                raise ValueError(f"Document {doc_id} not found")
                
            # Get topics for the document
            bow = self.corpus[doc_index]
            topics = self.lda_model.get_document_topics(bow)
            
            # Filter by threshold
            filtered_topics = [(topic_id, prob) for topic_id, prob in topics if prob >= threshold]
            
            return {
                'id': doc_id,
                'topics': filtered_topics,
                'metadata': self.documents[doc_index]['metadata']
            }
        else:
            # Get topics for all documents
            all_doc_topics = []
            for i, doc in enumerate(self.documents):
                bow = self.corpus[i]
                topics = self.lda_model.get_document_topics(bow)
                filtered_topics = [(topic_id, prob) for topic_id, prob in topics if prob >= threshold]
                
                all_doc_topics.append({
                    'id': doc['id'],
                    'topics': filtered_topics,
                    'metadata': doc['metadata']
                })
                
            return all_doc_topics
    
    def compute_coherence(self):
        """
        Compute coherence score for the model.
        
        Returns:
            float: Coherence score
        """
        if not self.lda_model:
            raise ValueError("Model not built yet")
            
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=[doc['tokens'] for doc in self.documents],
            dictionary=self.dictionary,
            coherence='c_v'
        )
        
        return coherence_model.get_coherence()
    
    def find_optimal_topics(self, start=2, limit=20, step=2):
        """
        Find the optimal number of topics using coherence scores.
        
        Args:
            start (int): Starting number of topics
            limit (int): Maximum number of topics
            step (int): Step size
            
        Returns:
            tuple: List of coherence scores and the optimal number of topics
        """
        if not self.documents:
            raise ValueError("No documents added to the model")
            
        coherence_scores = []
        models = []
        
        for num_topics in range(start, limit + 1, step):
            # Build model with current number of topics
            model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=10,
                alpha='auto',
                eta='auto',
                random_state=42
            )
            
            # Compute coherence score
            coherence_model = CoherenceModel(
                model=model,
                texts=[doc['tokens'] for doc in self.documents],
                dictionary=self.dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            models.append(model)
            
            print(f"Num topics: {num_topics}, Coherence score: {coherence_score}")
        
        # Find optimal number of topics
        optimal_index = np.argmax(coherence_scores)
        optimal_num_topics = start + optimal_index * step
        
        # Set the optimal model
        self.lda_model = models[optimal_index]
        
        return coherence_scores, optimal_num_topics
    
    def visualize_topics(self):
        """
        Create a visualization of topic distributions.
        
        Returns:
            matplotlib.figure.Figure: Figure object with the visualization
        """
        if not self.lda_model:
            raise ValueError("Model not built yet")
            
        # Get topic distributions for all documents
        doc_topics = self.get_document_topics()
        
        # Create a matrix of document-topic distributions
        num_topics = self.lda_model.num_topics
        topic_matrix = np.zeros((len(self.documents), num_topics))
        
        for i, doc in enumerate(doc_topics):
            for topic_id, prob in doc['topics']:
                topic_matrix[i, topic_id] = prob
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_matrix, cmap="YlGnBu", xticklabels=[f"Topic {i}" for i in range(num_topics)])
        plt.title("Document-Topic Distribution")
        plt.ylabel("Documents")
        plt.xlabel("Topics")
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_trending_topics(self, date_field='year'):
        """
        Identify trending topics over time.
        
        Args:
            date_field (str): Metadata field to use for dating documents
            
        Returns:
            dict: Trending topics over time
        """
        if not self.lda_model or not self.documents:
            raise ValueError("Model not built or no documents added")
            
        # Get document topics
        doc_topics = self.get_document_topics()
        
        # Group by date
        trend_data = defaultdict(lambda: [0] * self.lda_model.num_topics)
        date_counts = defaultdict(int)
        
        for doc in doc_topics:
            # Skip if date field not present
            if date_field not in doc['metadata']:
                continue
                
            date_value = doc['metadata'][date_field]
            date_counts[date_value] += 1
            
            # Add topic weights
            for topic_id, prob in doc['topics']:
                trend_data[date_value][topic_id] += prob
        
        # Normalize by document counts
        for date_value in trend_data:
            trend_data[date_value] = [weight / date_counts[date_value] for weight in trend_data[date_value]]
        
        # Convert to regular dict for return
        return {date: list(topics) for date, topics in trend_data.items()}
    
    def label_topics_automatically(self, num_words=5):
        """
        Automatically generate labels for topics.
        
        Args:
            num_words (int): Number of top words to use for labeling
            
        Returns:
            list: Topic labels
        """
        if not self.lda_model:
            raise ValueError("Model not built yet")
            
        labels = []
        
        for topic_id in range(self.lda_model.num_topics):
            # Get top words for the topic
            topic_words = self.lda_model.show_topic(topic_id, num_words)
            words = [word for word, _ in topic_words]
            
            # Create label from top words
            label = f"Topic {topic_id}: {', '.join(words)}"
            labels.append(label)
            
        return labels
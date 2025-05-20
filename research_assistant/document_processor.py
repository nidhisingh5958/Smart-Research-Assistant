"""
Document processing module for extracting text and metadata from research papers.
"""

import os
import re
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentProcessor:
    """
    Class for processing academic documents, primarily PDFs.
    Extracts text, metadata, sections, and references.
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Dictionary containing extracted text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "page_count": len(doc),
                "filename": os.path.basename(pdf_path)
            }
            
            # Extract text from all pages
            for page in doc:
                text += page.get_text()
                
            # Close the document
            doc.close()
            
            # Process extracted text
            processed_text = self._process_text(text)
            
            # Extract sections
            sections = self._extract_sections(processed_text)
            
            # Extract references
            references = self._extract_references(processed_text)
            
            return {
                "metadata": metadata,
                "full_text": processed_text,
                "sections": sections,
                "references": references
            }
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def _process_text(self, text):
        """
        Process extracted text by removing extra whitespace and normalizing.
        
        Args:
            text (str): Raw text extracted from PDF
            
        Returns:
            str: Processed text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove form feed characters
        text = text.replace('\f', '')
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _extract_sections(self, text):
        """
        Extract main sections from the paper.
        
        Args:
            text (str): Processed text
            
        Returns:
            dict: Dictionary of section names and their content
        """
        # Common section headers in research papers
        section_patterns = [
            r'Abstract', r'Introduction', r'Background', r'Related Work',
            r'Methodology', r'Methods', r'Experimental Setup', r'Experiments',
            r'Results', r'Discussion', r'Conclusion', r'References'
        ]
        
        # Create a regex pattern to match section headers
        pattern = r'(?:\n|^)(' + '|'.join(section_patterns) + r')(?:\n|\s)'
        
        # Find all section headers
        section_matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        # Extract sections
        sections = {}
        for i, match in enumerate(section_matches):
            section_name = match.group(1)
            start_pos = match.end()
            
            # If this is not the last section, get text until the next section
            if i < len(section_matches) - 1:
                end_pos = section_matches[i + 1].start()
                section_text = text[start_pos:end_pos].strip()
            else:
                # For the last section, get text until the end
                section_text = text[start_pos:].strip()
                
            sections[section_name] = section_text
            
        return sections
    
    def _extract_references(self, text):
        """
        Extract references from the paper.
        
        Args:
            text (str): Processed text
            
        Returns:
            list: List of extracted references
        """
        # Try to locate the references section
        references_section = None
        
        # Check for common headers for references section
        for header in ["References", "Bibliography", "Works Cited"]:
            pattern = re.compile(r'(?:\n|^)' + header + r'(?:\n|\s)', re.IGNORECASE)
            match = pattern.search(text)
            
            if match:
                references_section = text[match.end():].strip()
                break
        
        if not references_section:
            return []
        
        # Split references by looking for patterns like [1], 1., etc.
        ref_pattern = r'(?:\n|^)(?:\[\d+\]|\d+\.)'
        ref_matches = list(re.finditer(ref_pattern, references_section))
        
        references = []
        for i, match in enumerate(ref_matches):
            start_pos = match.start()
            
            # If this is not the last reference, get text until the next reference
            if i < len(ref_matches) - 1:
                end_pos = ref_matches[i + 1].start()
                reference_text = references_section[start_pos:end_pos].strip()
            else:
                # For the last reference, get text until the end
                reference_text = references_section[start_pos:].strip()
                
            references.append(reference_text)
            
        return references
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract keywords from text using frequency-based approach.
        
        Args:
            text (str): Text to extract keywords from
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of top keywords
        """
        # Tokenize text
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Calculate word frequencies
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
                
        # Sort by frequency
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [word for word, freq in sorted_keywords[:top_n]]
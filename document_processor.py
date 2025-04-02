import os
import re
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    A class for processing PDF documents, extracting text, and preparing it for analysis.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentProcessor with configuration parameters.
        
        Args:
            chunk_size: The size of text chunks for processing
            chunk_overlap: The overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_file: The PDF file object (from Streamlit file uploader)
            
        Returns:
            Tuple containing:
                - Full text of the document
                - Dictionary mapping page numbers to page text
        """
        pdf_reader = PdfReader(pdf_file)
        full_text = ""
        page_texts = {}
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_text = self._preprocess_text(page_text)
                page_texts[i+1] = page_text  # 1-indexed page numbers
                full_text += page_text + "\n\n"
        
        return full_text, page_texts
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = text.replace('|', 'I')  # Replace pipe with capital I
        text = text.replace('l', 'l')  # Replace lowercase L with lowercase L (font correction)
        
        # Remove header/footer patterns (customize based on document structure)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Clean up any remaining issues
        text = text.strip()
        
        return text
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Attempt to detect document sections based on common patterns.
        
        Args:
            text: Full document text
            
        Returns:
            Dictionary mapping section titles to section content
        """
        # Simple section detection based on numbered or capitalized headers
        # This is a basic implementation and may need customization for specific document formats
        section_pattern = r'(?:\n|^)(?:\d+\.\s+|[A-Z][A-Z\s]+:|\b[A-Z][A-Z\s]+\b)(.+?)(?=\n(?:\d+\.\s+|[A-Z][A-Z\s]+:|\b[A-Z][A-Z\s]+\b)|$)'
        
        sections = {}
        matches = re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            section_title = match.group(1).strip()
            section_content = match.group(0).strip()
            if len(section_title) > 3:  # Avoid very short section titles
                sections[section_title] = section_content
        
        return sections
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Document text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def process_document(self, pdf_file) -> Dict:
        """
        Process a PDF document completely - extract text, detect sections, and create chunks.
        
        Args:
            pdf_file: The PDF file object (from Streamlit file uploader)
            
        Returns:
            Dictionary containing processed document information
        """
        full_text, page_texts = self.extract_text_from_pdf(pdf_file)
        sections = self.detect_sections(full_text)
        chunks = self.split_text_into_chunks(full_text)
        
        return {
            "full_text": full_text,
            "page_texts": page_texts,
            "sections": sections,
            "chunks": chunks,
            "num_pages": len(page_texts),
            "num_chunks": len(chunks)
        }
    
    def extract_metadata(self, pdf_file) -> Dict:
        """
        Extract metadata from PDF document.
        
        Args:
            pdf_file: The PDF file object
            
        Returns:
            Dictionary of metadata
        """
        pdf_reader = PdfReader(pdf_file)
        metadata = pdf_reader.metadata
        
        result = {}
        if metadata:
            # Convert PDF metadata to dictionary
            for key in metadata:
                if metadata[key]:
                    # Clean the key name by removing the leading '/'
                    clean_key = key[1:] if key.startswith('/') else key
                    result[clean_key] = metadata[key]
        
        return result
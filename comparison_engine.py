import os
import time
import logging
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ComparisonEngine:
    """
    A class for comparing legal documents using RAG (Retrieval Augmented Generation).
    """
    
    def __init__(self, temperature: float = 0.0, persist_directory: str = "./chroma_db"):
        """
        Initialize the ComparisonEngine with necessary components.
        
        Args:
            temperature: Temperature setting for the LLM
            persist_directory: Directory to persist vector databases
        """
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4",  # Can be configured based on requirements
            request_timeout=120  # Add timeout of 120 seconds per request
        )
        
        # Set persistence directory
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Categories for comparison
        self.comparison_categories = [
            "Liability provisions",
            "Payment terms",
            "Termination conditions",
            "Warranty information",
            "Intellectual property rights",
            "Confidentiality requirements",
            "Dispute resolution mechanisms",
            "Force majeure clauses",
            "Notice requirements",
            "Amendment procedures"
        ]
    
    def create_vector_db(self, document_chunks: List[str], namespace: str, persist: bool = False) -> Chroma:
        """
        Create a vector database from document chunks.
        
        Args:
            document_chunks: List of text chunks from the document
            namespace: Namespace for the vector database (e.g., 'master' or 'client')
            persist: Whether to persist the vector database to disk
            
        Returns:
            Chroma vector database
        """
        persist_directory = f"{self.persist_directory}/{namespace}" if persist else None
        
        # Check if persistent DB already exists
        if persist and os.path.exists(persist_directory):
            logger.info(f"Loading existing vector database for {namespace}")
            return Chroma(
                embedding_function=self.embeddings,
                collection_name=f"document_{namespace}",
                persist_directory=persist_directory
            )
        
        logger.info(f"Creating new vector database for {namespace}")
        db = Chroma.from_texts(
            texts=document_chunks,
            embedding=self.embeddings,
            collection_name=f"document_{namespace}",
            persist_directory=persist_directory
        )
        
        if persist:
            logger.info(f"Persisting vector database for {namespace}")
            db.persist()
            
        return db
    
    def load_master_vector_db(self) -> Optional[Chroma]:
        """
        Load the master document vector database if it exists.
        
        Returns:
            Chroma vector database or None if it doesn't exist
        """
        master_dir = f"{self.persist_directory}/master"
        if os.path.exists(master_dir):
            logger.info("Loading existing master document vector database")
            return Chroma(
                embedding_function=self.embeddings,
                collection_name="document_master",
                persist_directory=master_dir
            )
        return None
    
    def delete_vector_db(self, namespace: str) -> bool:
        """
        Delete a persisted vector database.
        
        Args:
            namespace: Namespace of the vector database to delete
            
        Returns:
            True if successful, False otherwise
        """
        db_dir = f"{self.persist_directory}/{namespace}"
        if os.path.exists(db_dir):
            import shutil
            try:
                shutil.rmtree(db_dir)
                logger.info(f"Deleted vector database for {namespace}")
                return True
            except Exception as e:
                logger.error(f"Error deleting vector database: {str(e)}")
                return False
        return False
    
    def setup_retriever(self, vector_db: Chroma, k: int = 5) -> ContextualCompressionRetriever:
        """
        Set up a retriever with contextual compression.
        
        Args:
            vector_db: Vector database
            k: Number of documents to retrieve
            
        Returns:
            Configured retriever
        """
        # Base retriever
        base_retriever = vector_db.as_retriever(search_kwargs={"k": k})
        
        # Document compressor for extracting relevant information
        prompt_template = """
        Given the following document, extract the most relevant information related to legal terms and conditions:
        
        {context}
        
        Relevant information:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        
        compressor = LLMChainExtractor.from_llm(
            llm=self.llm,
            prompt=prompt
        )
        
        # Create a contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def compare_documents(self, 
                         master_retriever: ContextualCompressionRetriever,
                         client_retriever: ContextualCompressionRetriever) -> Dict[str, Any]:
        """
        Compare master and client documents across different categories.
        
        Args:
            master_retriever: Retriever for the master document
            client_retriever: Retriever for the client document
            
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        for category in self.comparison_categories:
            category_results = self._compare_category(
                category=category,
                master_retriever=master_retriever,
                client_retriever=client_retriever
            )
            results[category] = category_results
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(results)
        results["overall_summary"] = overall_summary
        
        return results
    
    def _compare_category(self,
                         category: str,
                         master_retriever: ContextualCompressionRetriever,
                         client_retriever: ContextualCompressionRetriever) -> Dict[str, Any]:
        """
        Compare documents for a specific category.
        
        Args:
            category: Category to compare
            master_retriever: Retriever for the master document
            client_retriever: Retriever for the client document
            
        Returns:
            Dictionary containing category comparison results
        """
        # Retrieve relevant sections from both documents
        logger.info(f"Retrieving master document sections for '{category}'")
        retrieval_start = time.time()
        master_docs = master_retriever.get_relevant_documents(category)
        logger.info(f"Retrieved {len(master_docs)} sections from master document in {time.time() - retrieval_start:.2f} seconds")
        
        retrieval_start = time.time()
        logger.info(f"Retrieving client document sections for '{category}'")
        client_docs = client_retriever.get_relevant_documents(category)
        logger.info(f"Retrieved {len(client_docs)} sections from client document in {time.time() - retrieval_start:.2f} seconds")
        
        # Extract text from retrieved documents
        master_text = "\n\n".join([doc.page_content for doc in master_docs])
        client_text = "\n\n".join([doc.page_content for doc in client_docs])
        
        # Create comparison prompt
        comparison_template = """
        You are a legal expert specializing in contract analysis. Compare the following sections from two documents regarding {category}.
        
        MASTER DOCUMENT:
        {master_text}
        
        CLIENT DOCUMENT:
        {client_text}
        
        Analyze these sections and identify:
        1. Discrepancies: Specific differences in terms or conditions
        2. Contradictions: Directly conflicting statements or requirements
        3. Missing clauses: Important elements present in one document but absent in the other
        4. Risk assessment: Potential legal or business risks arising from these differences
        5. Proposed solutions: For each identified issue, suggest a reasonable solution or compromise
        
        For each discrepancy, rate its severity:
        - MINOR: Small differences with minimal legal or business impact
        - MODERATE: Notable differences that should be addressed but aren't deal-breakers
        - MAJOR: Critical differences that present significant legal or business risks
        
        Format your response as a structured analysis with clear headings and bullet points.
        """
        
        comparison_prompt = PromptTemplate(
            template=comparison_template,
            input_variables=["category", "master_text", "client_text"]
        )
        
        # Create and run comparison chain
        comparison_chain = LLMChain(
            llm=self.llm,
            prompt=comparison_prompt
        )
        
        logger.info(f"Running LLM comparison for '{category}'")
        llm_start = time.time()
        try:
            response = comparison_chain.run(
                category=category,
                master_text=master_text,
                client_text=client_text
            )
            logger.info(f"LLM comparison completed in {time.time() - llm_start:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during LLM comparison: {str(e)}")
            response = f"Error during comparison: {str(e)}"
        
        # Extract discrepancies and proposed solutions
        discrepancies = self._extract_discrepancies(response)
        
        # Return structured results
        return {
            "analysis": response,
            "master_sections": master_text,
            "client_sections": client_text,
            "has_discrepancies": "discrepancies" in response.lower(),
            "has_contradictions": "contradictions" in response.lower(),
            "has_missing_clauses": "missing" in response.lower(),
            "discrepancies": discrepancies
        }
    
    def _extract_discrepancies(self, analysis: str) -> List[Dict[str, Any]]:
        """
        Extract structured discrepancies from the analysis text.
        
        Args:
            analysis: The analysis text from the LLM
            
        Returns:
            List of discrepancies with severity and proposed solutions
        """
        # This is a simple implementation - in a production system, you would use
        # a more sophisticated approach with the LLM to extract structured data
        discrepancies = []
        
        # Use LLM to extract structured discrepancies
        extraction_template = """
        Extract the discrepancies, their severity, and proposed solutions from the following analysis:
        
        {analysis}
        
        Format your response as a JSON array of objects, each with these fields:
        - description: A clear description of the discrepancy
        - severity: MINOR, MODERATE, or MAJOR
        - solution: The proposed solution for this discrepancy
        - category: The category this discrepancy belongs to (e.g., "Liability provisions")
        
        Response:
        """
        
        extraction_prompt = PromptTemplate(
            template=extraction_template,
            input_variables=["analysis"]
        )
        
        extraction_chain = LLMChain(
            llm=self.llm,
            prompt=extraction_prompt
        )
        
        try:
            extraction_result = extraction_chain.run(analysis=analysis)
            
            # Parse the JSON response - in a production system, add better error handling
            import json
            try:
                discrepancies = json.loads(extraction_result)
                if not isinstance(discrepancies, list):
                    discrepancies = []
            except:
                logger.error("Failed to parse discrepancies JSON")
                discrepancies = []
                
        except Exception as e:
            logger.error(f"Error extracting discrepancies: {str(e)}")
        
        return discrepancies
    
    def _generate_overall_summary(self, category_results: Dict[str, Any]) -> str:
        """
        Generate an overall summary of the comparison results.
        
        Args:
            category_results: Results from all category comparisons
            
        Returns:
            Overall summary text
        """
        # Create a summary of all categories
        categories_summary = ""
        
        for category, results in category_results.items():
            if any([results.get("has_discrepancies", False),
                   results.get("has_contradictions", False),
                   results.get("has_missing_clauses", False)]):
                categories_summary += f"- {category}: Has issues that need attention.\n"
            else:
                categories_summary += f"- {category}: No significant issues found.\n"
        
        # Create prompt for overall summary
        summary_template = """
        Based on the comparison of multiple categories in two legal documents, provide an executive summary of the findings.
        
        CATEGORY SUMMARIES:
        {categories_summary}
        
        Please provide:
        1. An overall assessment of the alignment between the documents
        2. The most critical areas requiring attention
        3. Recommended next steps for reconciliation
        
        Format your response as a concise executive summary suitable for business stakeholders.
        """
        
        summary_prompt = PromptTemplate(
            template=summary_template,
            input_variables=["categories_summary"]
        )
        
        # Create and run summary chain
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=summary_prompt
        )
        
        return summary_chain.run(categories_summary=categories_summary)

    def generate_annotated_pdf(self, 
                              pdf_path: str, 
                              discrepancies: List[Dict[str, Any]]) -> Tuple[str, bytes]:
        """
        Generate an annotated PDF with highlighted discrepancies.
        
        Args:
            pdf_path: Path to the original PDF
            discrepancies: List of discrepancies to highlight
            
        Returns:
            Tuple of (output_path, pdf_bytes)
        """
        if not discrepancies:
            logger.info("No discrepancies to annotate in PDF")
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            return pdf_path, pdf_bytes
            
        logger.info(f"Annotating PDF with {len(discrepancies)} discrepancies")
        
        # Define colors for different severity levels
        colors = {
            "MINOR": [0, 0.8, 0, 0.3],  # Green with 30% opacity
            "MODERATE": [1, 0.8, 0, 0.3],  # Yellow with 30% opacity
            "MAJOR": [1, 0, 0, 0.3]  # Red with 30% opacity
        }
        
        # Create output path for annotated PDF
        filename = os.path.basename(pdf_path)
        output_dir = os.path.dirname(pdf_path)
        output_path = os.path.join(output_dir, f"annotated_{filename}")
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract full text from the PDF
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += page.get_text()
            
            # Process each discrepancy
            for disc in discrepancies:
                description = disc.get('description', 'Discrepancy')
                severity = disc.get('severity', 'MINOR')
                solution = disc.get('solution', 'No solution provided')
                category = disc.get('category', 'Unknown category')
                
                # Use the client sections from the comparison results to find the exact text
                client_sections = disc.get('client_sections', '')
                
                # If we don't have client sections, use LLM to identify relevant paragraphs
                if not client_sections:
                    client_sections = self._identify_relevant_paragraphs(full_text, description)
                
                # Set highlight color based on severity
                highlight_color = colors.get(severity, colors["MINOR"])
                
                # Find and highlight the relevant paragraphs in the PDF
                self._highlight_paragraphs(doc, client_sections, highlight_color, 
                                          description, solution, category, severity)
            
            # Add a summary page at the end
            summary_page = doc.new_page(-1, width=doc[0].rect.width, height=doc[0].rect.height)
            
            # Create summary text
            summary_text = "# T&C Sentinel - Discrepancy Summary\n\n"
            
            # Group discrepancies by category
            discrepancies_by_category = {}
            for disc in discrepancies:
                category = disc.get('category', 'Uncategorized')
                if category not in discrepancies_by_category:
                    discrepancies_by_category[category] = []
                discrepancies_by_category[category].append(disc)
            
            # Add each category and its discrepancies
            for category, discs in discrepancies_by_category.items():
                summary_text += f"## {category}\n\n"
                for disc in discs:
                    severity = disc.get('severity', 'MINOR')
                    description = disc.get('description', 'Discrepancy')
                    solution = disc.get('solution', 'No solution provided')
                    
                    summary_text += f"* [{severity}] {description}\n"
                    summary_text += f"  - Proposed solution: {solution}\n\n"
            
            # Add the summary text to the page
            summary_page.insert_text((72, 72), summary_text, fontsize=11)
            
            # Save the annotated PDF
            doc.save(output_path)
            doc.close()
            
            # Read the annotated PDF
            with open(output_path, 'rb') as f:
                pdf_bytes = f.read()
                
            logger.info(f"Successfully created annotated PDF at {output_path}")
            return output_path, pdf_bytes
            
        except Exception as e:
            logger.error(f"Error annotating PDF: {str(e)}")
            # Return original PDF if annotation fails
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            return pdf_path, pdf_bytes
    
    def _identify_relevant_paragraphs(self, full_text: str, description: str) -> str:
        """
        Use LLM to identify relevant paragraphs in the document that match the discrepancy.
        
        Args:
            full_text: Full text of the document
            description: Description of the discrepancy
            
        Returns:
            Relevant paragraphs from the document
        """
        # Create a prompt for the LLM to identify relevant paragraphs
        identification_template = """
        I need to identify the exact paragraphs in a legal document that relate to this discrepancy:
        
        DISCREPANCY: {description}
        
        Here is the document text:
        {text}
        
        Please extract ONLY the exact paragraphs or clauses from the document that directly relate to this discrepancy.
        Do not include any analysis or commentary. Return ONLY the exact text from the document.
        
        RELEVANT PARAGRAPHS:
        """
        
        identification_prompt = PromptTemplate(
            template=identification_template,
            input_variables=["description", "text"]
        )
        
        identification_chain = LLMChain(
            llm=self.llm,
            prompt=identification_prompt
        )
        
        try:
            # If the text is too long, we need to chunk it
            if len(full_text) > 12000:  # LLM context limit
                # Simple chunking - in a production system, use more sophisticated chunking
                chunks = [full_text[i:i+12000] for i in range(0, len(full_text), 12000)]
                
                all_paragraphs = []
                for chunk in chunks:
                    result = identification_chain.run(
                        description=description,
                        text=chunk
                    )
                    if result and len(result) > 20:  # Only include meaningful results
                        all_paragraphs.append(result)
                
                return "\n\n".join(all_paragraphs)
            else:
                return identification_chain.run(
                    description=description,
                    text=full_text
                )
        except Exception as e:
            logger.error(f"Error identifying relevant paragraphs: {str(e)}")
            return ""
    
    def _highlight_paragraphs(self, doc, text_to_highlight: str, color, 
                             description: str, solution: str, category: str, severity: str):
        """
        Find and highlight paragraphs in the PDF.
        
        Args:
            doc: PDF document
            text_to_highlight: Text to find and highlight
            color: Highlight color
            description: Discrepancy description
            solution: Proposed solution
            category: Discrepancy category
            severity: Discrepancy severity
        """
        if not text_to_highlight or len(text_to_highlight) < 20:
            logger.warning("Text to highlight is too short or empty")
            return
            
        # Clean up the text to highlight (remove extra whitespace, normalize line breaks)
        text_to_highlight = re.sub(r'\s+', ' ', text_to_highlight).strip()
        
        # Break into paragraphs for more precise matching
        paragraphs = [p for p in text_to_highlight.split('\n') if len(p) > 20]
        if not paragraphs:
            paragraphs = [text_to_highlight]
        
        # Comment to add to highlights
        comment = f"[{severity}] {description}\n\nProposed solution: {solution}"
        
        # Search each page for the paragraphs
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            for paragraph in paragraphs:
                # Clean up paragraph
                paragraph = re.sub(r'\s+', ' ', paragraph).strip()
                if len(paragraph) < 20:
                    continue
                
                # Try to find the paragraph in the page
                if paragraph in page_text:
                    # Find all instances of the paragraph
                    instances = page.search_for(paragraph)
                    
                    # Highlight each instance
                    for inst in instances:
                        # Add highlight
                        annot = page.add_highlight_annot(inst)
                        annot.set_colors(stroke=color)
                        annot.update()
                        
                        # Add comment
                        annot = page.add_text_annot(inst.br, comment)
                        annot.set_info(title=f"T&C Sentinel - {category}")
                        annot.update()
                else:
                    # If exact match fails, try to find key sentences
                    sentences = [s for s in paragraph.split('.') if len(s) > 15]
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) < 15:
                            continue
                            
                        if sentence in page_text:
                            instances = page.search_for(sentence)
                            
                            for inst in instances:
                                annot = page.add_highlight_annot(inst)
                                annot.set_colors(stroke=color)
                                annot.update()
                                
                                annot = page.add_text_annot(inst.br, comment)
                                annot.set_info(title=f"T&C Sentinel - {category}")
                                annot.update()

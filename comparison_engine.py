import os
import time
import logging
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Try to import optional dependencies
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"OpenAI embeddings not available: {str(e)}")
    OPENAI_AVAILABLE = False

try:
    from langchain.chat_models import ChatOpenAI
    CHATGPT_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"ChatGPT not available: {str(e)}")
    CHATGPT_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Transformers not available: {str(e)}")
    TRANSFORMERS_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.warning(f"Google Gemini not available: {str(e)}")
    GEMINI_AVAILABLE = False

# Import the rest of the dependencies
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings.base import Embeddings

class HuggingFaceEmbeddings(Embeddings):
    """
    A class to provide embeddings using Hugging Face's transformers library directly.
    This avoids the need for sentence-transformers and uses the transformers package
    that's already installed for the Saul model.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFaceEmbeddings with the specified model.
        
        Args:
            model_name: Name of the Hugging Face model to use for embeddings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for HuggingFaceEmbeddings")
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Initialized Hugging Face embeddings model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face embeddings model: {str(e)}")
            raise RuntimeError(f"Failed to initialize embeddings model: {str(e)}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding for the text
        """
        # Tokenize and prepare for the model
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling - take average of all token embeddings
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Convert to list and return
        return mean_embeddings.cpu().numpy()[0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embeddings, one for each document
        """
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding for the query
        """
        return self._get_embedding(text)

class ModelProvider:
    """
    A class to provide different LLM models for document comparison.
    """
    
    def __init__(self, model_type: str = "openai", temperature: float = 0.0):
        """
        Initialize the ModelProvider with the specified model type.
        
        Args:
            model_type: Type of model to use ('openai', 'gemini', or 'saul')
            temperature: Temperature setting for the LLM
        """
        self.model_type = model_type
        self.temperature = temperature
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on model_type."""
        if self.model_type == "openai" and CHATGPT_AVAILABLE:
            try:
                # Check for API key
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                    
                self.llm = ChatOpenAI(
                    temperature=self.temperature,
                    model_name="gpt-4",  # Can be configured based on requirements
                    request_timeout=120,  # Add timeout of 120 seconds per request
                    openai_api_key=api_key
                )
                logger.info("Initialized OpenAI GPT-4 model")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI model: {str(e)}")
                self._fallback_to_available_model()
        elif self.model_type == "gemini" and GEMINI_AVAILABLE:
            try:
                # Check for API key
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                    
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=self.temperature,
                    google_api_key=api_key,
                    timeout=120
                )
                logger.info("Initialized Google Gemini model")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {str(e)}")
                self._fallback_to_available_model()
        elif self.model_type == "saul" and TRANSFORMERS_AVAILABLE:
            self._initialize_saul_model()
        else:
            logger.warning(f"Unknown or unavailable model type: {self.model_type}")
            self._fallback_to_available_model()
    
    def _fallback_to_available_model(self):
        """Fallback to any available model."""
        if CHATGPT_AVAILABLE:
            try:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    logger.info("Falling back to OpenAI GPT-4 model")
                    self.model_type = "openai"
                    self.llm = ChatOpenAI(
                        temperature=self.temperature,
                        model_name="gpt-4",
                        request_timeout=120,
                        openai_api_key=api_key
                    )
                    return
            except Exception:
                pass
                
        if GEMINI_AVAILABLE:
            try:
                api_key = os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    logger.info("Falling back to Google Gemini model")
                    self.model_type = "gemini"
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.0-pro",
                        temperature=self.temperature,
                        google_api_key=api_key,
                        timeout=120
                    )
                    return
            except Exception:
                pass
                
        if TRANSFORMERS_AVAILABLE:
            logger.info("Falling back to Saul model")
            self.model_type = "saul"
            self._initialize_saul_model()
            return
            
        raise RuntimeError("No available models to use. Please ensure at least one of the API keys (OPENAI_API_KEY or GOOGLE_API_KEY) is set or that the transformers library is installed for the Saul model.")
    
    def _initialize_saul_model(self):
        """Initialize the Saul model pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for Saul model")
            
        try:
            # Initialize the Saul model pipeline
            model_name = "Equall/Saul-Instruct-v1"
            
            # Check if model exists locally first
            try:
                self.pipeline = pipeline(
                    "text-generation", 
                    model=model_name, 
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
            except Exception:
                # If model doesn't exist locally, try to download with specific parameters
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    repo_id=model_name,
                    local_files_only=False,
                    resume_download=True
                )
                
                self.pipeline = pipeline(
                    "text-generation", 
                    model=model_path, 
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                
            logger.info("Initialized Saul-Instruct-v1 model")
        except Exception as e:
            logger.error(f"Failed to initialize Saul model: {str(e)}")
            self._fallback_to_available_model()
    
    def run_chain(self, prompt: PromptTemplate, **kwargs) -> str:
        """
        Run the LLM chain with the given prompt and kwargs.
        
        Args:
            prompt: The prompt template to use
            **kwargs: The keyword arguments to pass to the prompt
            
        Returns:
            The generated text
        """
        if self.model_type == "openai":
            # Use LangChain for OpenAI
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**kwargs)
        elif self.model_type == "gemini":
            # Use LangChain for Gemini
            chain = LLMChain(llm=self.llm, prompt=prompt)
            return chain.run(**kwargs)
        elif self.model_type == "saul":
            # Format the prompt for Saul model
            formatted_prompt = prompt.format(**kwargs)
            
            # Create a chat message format
            messages = [
                {"role": "user", "content": formatted_prompt},
            ]
            
            # Apply chat template
            prompt_text = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate response
            outputs = self.pipeline(
                prompt_text, 
                max_new_tokens=1024,  # Adjust as needed
                do_sample=False
            )
            
            # Extract the generated text, removing the prompt
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt part from the generated text
            response = generated_text[len(prompt_text):].strip()
            
            return response
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def get_llm(self):
        """
        Get the LLM instance for use with LangChain components.
        Only applicable for OpenAI and Gemini models.
        
        Returns:
            The LLM instance
        """
        if self.model_type in ["openai", "gemini"]:
            return self.llm
        else:
            raise ValueError(f"Direct LLM access not available for model type: {self.model_type}")

class ComparisonEngine:
    """
    A class for comparing legal documents using RAG (Retrieval Augmented Generation).
    """
    
    def __init__(self, temperature: float = 0.0, persist_directory: str = "./chroma_db", model_type: str = "openai", embedding_model: str = "openai"):
        """
        Initialize the ComparisonEngine with necessary components.
        
        Args:
            temperature: Temperature setting for the LLM
            persist_directory: Directory to persist vector databases
            model_type: Type of model to use for comparison ('openai', 'gemini', or 'saul')
            embedding_model: Type of model to use for embeddings ('openai', 'gemini', or 'huggingface')
        """
        # Store the embedding model type for later reference
        self.embedding_model_type = embedding_model
        
        # Set persistence directory
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings based on the specified model
        self._initialize_embeddings(embedding_model)
        
        # Initialize model provider
        self.model_provider = ModelProvider(model_type=model_type, temperature=temperature)
        
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
    
    def _initialize_embeddings(self, embedding_model: str):
        """
        Initialize embeddings based on the specified model.
        
        Args:
            embedding_model: Type of model to use for embeddings ('openai', 'gemini', or 'huggingface')
        """
        if embedding_model == "openai" and OPENAI_AVAILABLE:
            try:
                # Check for API key
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                    
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                logger.info("Using OpenAI embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {str(e)}")
        
        if embedding_model == "gemini" and GEMINI_AVAILABLE:
            try:
                # Check for API key
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                    
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                logger.info("Using Google Gemini embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini embeddings: {str(e)}")
        
        # Fall back to HuggingFace embeddings if other options fail
        self._initialize_local_embeddings()
    
    def _initialize_local_embeddings(self):
        """Initialize local embeddings when cloud-based options are not available."""
        try:
            # First try to use sentence-transformers if available
            try:
                from langchain.embeddings import HuggingFaceEmbeddings as LangchainHFEmbeddings
                
                self.embeddings = LangchainHFEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
                logger.info("Using sentence-transformers embeddings (via LangChain)")
                return
            except ImportError:
                pass
                
            # Fall back to our custom implementation
            self.embeddings = HuggingFaceEmbeddings()
            logger.info("Using custom HuggingFace embeddings implementation")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {str(e)}")
            raise RuntimeError(f"No embedding models available: {str(e)}")
    
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
        # Check if we need to recreate the database due to embedding model change
        embedding_model_file = os.path.join(self.persist_directory, f"{namespace}_embedding_model.txt")
        
        # If the vector database exists, check if it was created with a different embedding model
        if persist and os.path.exists(os.path.join(self.persist_directory, namespace)) and os.path.exists(embedding_model_file):
            with open(embedding_model_file, 'r') as f:
                stored_model = f.read().strip()
                
            # If the embedding model has changed, we need to delete the existing database
            if stored_model != self.embedding_model_type:
                logger.info(f"Embedding model changed from {stored_model} to {self.embedding_model_type}. Recreating vector database.")
                self.delete_vector_db(namespace)
        
        # Import ChromaDB client here to ensure proper initialization
        from chromadb.config import Settings
        import chromadb
        
        # Create the vector database
        if persist:
            # Create a persistent vector database with explicit client settings
            chroma_client = chromadb.PersistentClient(
                path=os.path.join(self.persist_directory, namespace),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            try:
                collection = chroma_client.get_or_create_collection(name=namespace)
            except Exception as e:
                logger.warning(f"Error getting collection, trying to create new: {str(e)}")
                # If collection exists but is corrupted, try to delete and recreate
                try:
                    chroma_client.delete_collection(name=namespace)
                    collection = chroma_client.create_collection(name=namespace)
                except Exception as inner_e:
                    logger.error(f"Failed to recreate collection: {str(inner_e)}")
                    raise
            
            # Add documents to collection
            ids = [str(i) for i in range(len(document_chunks))]
            embeddings = self.embeddings.embed_documents(document_chunks)
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(document_chunks), batch_size):
                end_idx = min(i + batch_size, len(document_chunks))
                collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=document_chunks[i:end_idx],
                    metadatas=[{"source": namespace} for _ in range(i, end_idx)]
                )
            
            # Create LangChain Chroma wrapper
            vector_db = Chroma(
                client=chroma_client,
                collection_name=namespace,
                embedding_function=self.embeddings
            )
            
            # Save the embedding model type
            with open(embedding_model_file, 'w') as f:
                f.write(self.embedding_model_type)
                
            logger.info(f"Created persistent vector database for {namespace} using {self.embedding_model_type} embeddings")
        else:
            # Create an in-memory vector database
            chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            # Create collection
            collection = chroma_client.create_collection(name=namespace)
            
            # Add documents to collection
            ids = [str(i) for i in range(len(document_chunks))]
            embeddings = self.embeddings.embed_documents(document_chunks)
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(document_chunks), batch_size):
                end_idx = min(i + batch_size, len(document_chunks))
                collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=document_chunks[i:end_idx],
                    metadatas=[{"source": namespace} for _ in range(i, end_idx)]
                )
            
            # Create LangChain Chroma wrapper
            vector_db = Chroma(
                client=chroma_client,
                collection_name=namespace,
                embedding_function=self.embeddings
            )
            
            logger.info(f"Created in-memory vector database for {namespace} using {self.embedding_model_type} embeddings")
        
        return vector_db
    
    def load_master_vector_db(self):
        """
        Load the master document vector database if it exists.
        
        Returns:
            Chroma vector database or None if it doesn't exist
        """
        master_db_path = os.path.join(self.persist_directory, "master")
        embedding_model_file = os.path.join(self.persist_directory, "master_embedding_model.txt")
        
        if os.path.exists(master_db_path) and os.path.isdir(master_db_path):
            # Check if the embedding model has changed
            if os.path.exists(embedding_model_file):
                with open(embedding_model_file, 'r') as f:
                    stored_model = f.read().strip()
                
                if stored_model != self.embedding_model_type:
                    logger.warning(f"Master database was created with {stored_model} embeddings, but current model is {self.embedding_model_type}.")
                    logger.warning("Cannot load master database with different embedding dimensions. Please recreate the master database.")
                    return None
            
            try:
                # Import ChromaDB client here to ensure proper initialization
                from chromadb.config import Settings
                import chromadb
                
                # Create client with explicit settings
                chroma_client = chromadb.PersistentClient(
                    path=master_db_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # Create Chroma wrapper
                vector_db = Chroma(
                    client=chroma_client,
                    collection_name="master",
                    embedding_function=self.embeddings
                )
                
                logger.info("Loaded master vector database")
                return vector_db
            except Exception as e:
                logger.error(f"Failed to load master vector database: {str(e)}")
                return None
        else:
            logger.info("No master vector database found")
            return None
    
    def delete_vector_db(self, namespace: str):
        """
        Delete a persisted vector database.
        
        Args:
            namespace: Namespace of the vector database to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db_path = os.path.join(self.persist_directory, namespace)
            embedding_model_file = os.path.join(self.persist_directory, f"{namespace}_embedding_model.txt")
            
            # Delete the vector database directory if it exists
            if os.path.exists(db_path) and os.path.isdir(db_path):
                import shutil
                shutil.rmtree(db_path)
                logger.info(f"Deleted vector database: {namespace}")
            
            # Delete the embedding model file if it exists
            if os.path.exists(embedding_model_file):
                os.remove(embedding_model_file)
                logger.info(f"Deleted embedding model file for: {namespace}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector database {namespace}: {str(e)}")
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
        # Create a base retriever
        base_retriever = vector_db.as_retriever(search_kwargs={"k": k})
        
        # Document compressor for extracting relevant information
        prompt_template = """
        Given the following document, extract the most relevant information related to legal terms and conditions:
        
        {context}
        
        Relevant information:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        
        # Use the appropriate LLM based on model type
        if self.model_provider.model_type in ["openai", "gemini"]:
            try:
                # Try to use the standard LLMChainExtractor
                compressor = LLMChainExtractor.from_llm(
                    llm=self.model_provider.get_llm(),
                    prompt=prompt
                )
                
                # Create a contextual compression retriever
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                
                return compression_retriever
            except Exception as e:
                logger.warning(f"Failed to create standard compressor: {str(e)}")
                logger.info("Falling back to simple retriever without compression")
                return base_retriever
        else:
            # For models that don't support the standard compressor, just return the base retriever
            logger.info(f"Using simple retriever without compression for {self.model_provider.model_type} model")
            return base_retriever
    
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
        
        Also, for each discrepancy, classify it as:
        - ACTION_NEEDED: The discrepancy needs to be addressed (e.g., client document contradicts master document in a way that creates risk)
        - ACCEPTABLE: The discrepancy is acceptable (e.g., master document has provisions that client document lacks, but this is beneficial to the company)
        
        When determining if a discrepancy is ACCEPTABLE or ACTION_NEEDED, consider:
        - If master document has stronger protections that are missing from client document, this is likely ACTION_NEEDED
        - If master document has provisions that benefit the company and are missing from client document, this is likely ACTION_NEEDED
        - If client document has additional obligations not in master document, this is likely ACTION_NEEDED
        - If client document has weaker language than master document, this is likely ACTION_NEEDED
        - If master document has termination rights, liability limitations, or other protective clauses missing from client document, this may be ACCEPTABLE if it's beneficial to maintain these differences
        
        Format your response as a structured analysis with clear headings and bullet points.
        """
        
        comparison_prompt = PromptTemplate(
            template=comparison_template,
            input_variables=["category", "master_text", "client_text"]
        )
        
        # Run comparison using the model provider
        logger.info(f"Running LLM comparison for '{category}'")
        llm_start = time.time()
        try:
            response = self.model_provider.run_chain(
                comparison_prompt,
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
        Extract the discrepancies, their severity, proposed solutions, and action classification from the following analysis:
        
        {analysis}
        
        Format your response as a JSON array of objects, each with these fields:
        - description: A clear description of the discrepancy
        - severity: MINOR, MODERATE, or MAJOR
        - solution: The proposed solution for this discrepancy
        - category: The category this discrepancy belongs to (e.g., "Liability provisions")
        - action_needed: true if the discrepancy needs to be addressed, false if it's acceptable as is
        - rationale: A brief explanation of why action is or is not needed
        
        Response:
        """
        
        extraction_prompt = PromptTemplate(
            template=extraction_template,
            input_variables=["analysis"]
        )
        
        try:
            extraction_result = self.model_provider.run_chain(extraction_prompt, analysis=analysis)
            
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
        
        try:
            summary = self.model_provider.run_chain(summary_prompt, categories_summary=categories_summary)
        except Exception as e:
            logger.error(f"Error generating overall summary: {str(e)}")
            summary = "Error generating summary. Please review individual category analyses."
            
        return summary
    
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
            llm=self.model_provider.get_llm(),
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

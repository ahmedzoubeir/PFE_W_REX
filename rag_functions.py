import os
import tempfile
import traceback
import asyncio
import hashlib
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# PDF, text, and data processing imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    
)
#from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Vector storage and embeddings
import chromadb
from chromadb.config import Settings
import ollama
from rank_bm25 import BM25Okapi

# LLM chain components
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'xlsx'}
UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_files")
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")

class KeywordIndex:
    """Simple keyword index using BM25 for efficient keyword matching"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.tokenized_corpus = []
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index a list of documents for keyword search
        
        Args:
            documents: List of document dictionaries with 'content' and 'id' fields
        """
        self.documents = documents
        self.doc_ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        
        # Tokenize and create corpus
        tokenized_docs = []
        for doc in documents:
            content = doc.get('content', '')
            # Simple tokenization (can be improved)
            tokens = re.findall(r'\w+', content.lower())
            tokenized_docs.append(tokens)
        
        self.tokenized_corpus = tokenized_docs
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
    async def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search documents using BM25 keyword search
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with scores
        """
        if not self.bm25:
            return []
            
        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair scores with documents and sort
        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc, score in scored_docs[:top_k]:
            results.append({
                'document': doc,
                'score': float(score),
                'source': 'keyword'
            })
            
        return results

class HybridRetrievalOrchestrator:
    """Orchestrates hybrid search across multiple retrieval methods"""
    
    def __init__(self, vector_store, keyword_index):
        """
        Initialize with retrieval components
        
        Args:
            vector_store: ChromaDB collection or similar vector store
            keyword_index: KeywordIndex instance
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.embed_model = "nomic-embed-text:latest"  # Default embedding model
        
    def set_embed_model(self, model_name: str):
        """Set the embedding model to use"""
        self.embed_model = model_name
        
    async def _semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search using vector embeddings
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with scores
        """
        try:
            # Generate embedding for the query
            query_response = ollama.embed(model=self.embed_model, input=query)
            query_embedding = query_response["embeddings"]
            
            # Ensure embeddings are in the correct format
            if isinstance(query_embedding, list) and len(query_embedding) == 1 and isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
            
            # Search the vector store
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            semantic_results = []
            for i in range(len(results["documents"][0])):
                semantic_results.append({
                    'document': {
                        'id': results["ids"][0][i],
                        'content': results["documents"][0][i],
                    },
                    'score': float(results["distances"][0][i]) if "distances" in results else 1.0,
                    'source': 'semantic'
                })
                
            return semantic_results
            
        except Exception as e:
            print(f"Semantic search error: {str(e)}")
            return []
    
    async def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform keyword-based search
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of document dictionaries with scores
        """
        try:
            return await self.keyword_index.search(query, top_k)
        except Exception as e:
            print(f"Keyword search error: {str(e)}")
            return []
    
    def _combine_and_rank(self, 
                         vector_results: List[Dict], 
                         keyword_results: List[Dict]) -> List[Dict]:
        """
        Combine and rank results from different search methods
        
        Args:
            vector_results: Results from semantic search
            keyword_results: Results from keyword search
            
        Returns:
            Ranked list of combined results
        """
        all_results = []
        seen_docs = set()
        
        # Helper function to normalize scores within a method
        def normalize_scores(results):
            if not results:
                return results
                
            # Get min and max scores
            scores = [r['score'] for r in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            # Normalize scores to 0-1 range
            for r in results:
                if score_range > 0:
                    r['score'] = (r['score'] - min_score) / score_range
                else:
                    r['score'] = 1.0
                    
            return results
        
        # Normalize scores for each method
        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)
        
        # Combine semantic and keyword results
        for result in vector_results:
            doc_id = result['document'].get('id', '')
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                result['methods'] = ['semantic']
                result['combined_score'] = result['score'] * 0.7  # Weight semantic results higher
                all_results.append(result)
        
        for result in keyword_results:
            doc_id = result['document'].get('id', '')
            if doc_id in seen_docs:
                # Document already added from semantic search, update it
                for existing in all_results:
                    if existing['document'].get('id', '') == doc_id:
                        existing['methods'].append('keyword')
                        # Boost score with keyword match
                        existing['combined_score'] = existing['combined_score'] + (result['score'] * 0.3)
                        break
            else:
                # New document from keyword search
                seen_docs.add(doc_id)
                result['methods'] = ['keyword']
                result['combined_score'] = result['score'] * 0.3  # Lower weight for keyword-only matches
                all_results.append(result)
        
        # Sort by combined score
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return all_results
    
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve documents using hybrid search approach
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Ranked list of relevant documents
        """
        # Execute search methods concurrently
        vector_results, keyword_results = await asyncio.gather(
            self._semantic_search(query, top_k=top_k),
            self._keyword_search(query, top_k=top_k)
        )
        
        # Combine and rank results
        combined = self._combine_and_rank(vector_results, keyword_results)
        
        return combined[:top_k]

class HybridReasoningModule:
    """Analyzes and reasons over hybrid search results to extract insights"""
    
    def __init__(self, llm_model="llama2:13b"):
        """
        Initialize with an LLM model
        
        Args:
            llm_model: Name of the Ollama model to use
        """
        self.llm_model = llm_model
        
    def _align_sources(self, combined_docs: List[Dict]) -> List[Dict]:
        """
        Align information across different search methods
        
        Args:
            combined_docs: Combined search results
            
        Returns:
            List of aligned facts with confidence scores
        """
        aligned_facts = []
        
        # Group documents by content similarity
        content_groups = {}
        
        for doc in combined_docs:
            content = doc['document'].get('content', '')
            methods = doc.get('methods', [])
            
            # Create a simple hash of content for grouping
            content_hash = hash(content[:100])  # Using first 100 chars as approximation
            
            if content_hash not in content_groups:
                content_groups[content_hash] = {
                    'content': content,
                    'methods': set(methods),
                    'score': doc.get('combined_score', 0),
                    'count': 1
                }
            else:
                # Update existing group
                content_groups[content_hash]['methods'].update(methods)
                content_groups[content_hash]['score'] += doc.get('combined_score', 0)
                content_groups[content_hash]['count'] += 1
        
        # Calculate confidence based on method diversity and scores
        for group_id, group in content_groups.items():
            # Higher confidence if found by multiple methods
            method_diversity = len(group['methods']) / 2  # Normalize (max 2 methods)
            # Higher confidence if high score
            score_factor = group['score'] / group['count']
            # Combined confidence
            confidence = (method_diversity * 0.5) + (score_factor * 0.5)
            
            aligned_facts.append({
                'content': group['content'],
                'methods': list(group['methods']),
                'confidence': min(confidence, 1.0),  # Cap at 1.0
                'support_count': group['count']
            })
        
        # Sort by confidence
        aligned_facts.sort(key=lambda x: x['confidence'], reverse=True)
        
        return aligned_facts
    
    async def process_context(self, combined_docs: List[Dict]) -> Dict:
        """
        Process and analyze combined documents to extract insights
        
        Args:
            combined_docs: List of combined search results
            
        Returns:
            Dictionary of insights with confidence scores
        """
        # Align information from different sources
        aligned_facts = self._align_sources(combined_docs)
        
        return {"insights": aligned_facts}

def prepare_context_from_hybrid_results(hybrid_results):
    """
    Prepare context from hybrid search results for the LLM
    
    Args:
        hybrid_results: Results from hybrid search
        
    Returns:
        String context for the LLM
    """
    contexts = []
    
    for i, result in enumerate(hybrid_results):
        document = result.get('document', {})
        content = document.get('content', '')
        methods = result.get('methods', [])
        score = result.get('combined_score', 0)
        
        # Add relevance information
        contexts.append(f"[Document {i+1} - Relevance: {score:.2f}, Method: {', '.join(methods)}]\n{content}\n")
    
    return "\n\n".join(contexts)

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    """Get the extension of a file"""
    return filename.rsplit('.', 1)[1].lower()

def process_document(file_path, file_extension):
    """
    Process a document and split it into chunks
    
    Args:
        file_path: Path to the document
        file_extension: Extension of the file
        
    Returns:
        List of document texts
    """
    # Load document based on file extension
    if file_extension == 'pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_extension == 'txt':
        loader = TextLoader(file_path)
        docs = loader.load()
    elif file_extension == 'csv':
        loader = CSVLoader(file_path)
        docs = loader.load()
    elif file_extension == 'xlsx':
        loader = UnstructuredLoader(file_path)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
        
    print(f"Starting text splitting on {len(docs)} documents")
        
    # Apply normal text splitting for all file types
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=290,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(docs)
    print(f"Split into {len(documents)} chunks")

    if not documents:
        raise ValueError("No text extracted from document.")
        
    # Convert documents to simple strings as per Ollama's examples
    document_texts = [doc.page_content for doc in documents]
        
    # Return document texts
    return document_texts

def create_vector_store(documents, model_choice, filename, recreate=True):
    """
    Create vector store following Ollama's exact approach with batching
    
    Args:
        documents: List of document texts
        model_choice: Ollama embedding model to use
        filename: Name of the file
        recreate: Whether to recreate the collection
        
    Returns:
        bool: Success status
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
        
    # Handle collection creation/reuse
    if recreate:
        try:
            client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception as e:
            print(f"No collection to delete or error: {str(e)}")
            
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection '{collection_name}'")
    else:
        # Try to get existing collection, create if it doesn't exist
        try:
            collection = client.get_collection(name=collection_name)
            print(f"Using existing collection '{collection_name}'")
        except:
            collection = client.create_collection(name=collection_name)
            print(f"Created new collection '{collection_name}' as none existed")
    
    # Process in smaller batches to avoid overloading the server
    batch_size = 25  # Reduced batch size for better stability
    total_docs = len(documents)
        
    print(f"Starting embedding process for {total_docs} documents in batches of {batch_size}")
        
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        current_batch = documents[batch_start:batch_end]
            
        print(f"Processing batch {batch_start//batch_size + 1}: documents {batch_start} to {batch_end-1}")
            
        # Process each document in the current batch
        for i, d in enumerate(current_batch):
            global_idx = batch_start + i
            try:
                # Limit content size if necessary to avoid issues with very large documents
                content = d[:8000] if len(d) > 8000 else d
                    
                # Skip empty content
                if not content.strip():
                    print(f"Skipping empty document {global_idx}")
                    continue
                    
                response = ollama.embed(model=model_choice, input=content)
                embeddings = response["embeddings"]
                    
                # Add to collection
                collection.add(
                    ids=[str(global_idx)],
                    embeddings=embeddings,
                    documents=[content]
                )
            except Exception as e:
                print(f"Error embedding document {global_idx}: {str(e)}")
                # Continue with next document instead of failing the whole batch
                continue
            
        print(f"Completed batch {batch_start//batch_size + 1}")
        
    print(f"Successfully added documents to vector store. Total count: {collection.count()}")
    return True


# Add these functions to your rag_functions.py file

def extract_file_content(file_path):
    """
    Extract raw text content from a file for comparison purposes
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Raw text content of the file
    """
    try:
        file_extension = get_file_extension(file_path)
        
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            content = " ".join([doc.page_content for doc in docs])
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif file_extension == 'csv':
            df = pd.read_csv(file_path)
            content = df.to_string()
        elif file_extension == 'xlsx':
            # We'll need a simplified approach for XLSX files since we removed UnstructuredLoader
            df = pd.read_excel(file_path)
            content = df.to_string()
        else:
            return ""
            
        return content
    except Exception as e:
        print(f"Error extracting content from {file_path}: {str(e)}")
        return ""


def get_content_hash(content):
    """
    Generate a hash of the file content
    
    Args:
        content: File content as string
        
    Returns:
        str: Hash of the content
    """
    # Use SHA-256 for reliable hashing
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def find_similar_file(new_file_path, similarity_threshold=0.9):
    """
    Check if a similar file exists in the upload directory
    
    Args:
        new_file_path: Path to the new file
        similarity_threshold: Threshold for considering files similar (0.0-1.0)
        
    Returns:
        tuple: (bool, str) - (is_duplicate, collection_name if duplicate)
    """
    try:
        # Extract content from new file
        new_content = extract_file_content(new_file_path)
        if not new_content:
            return False, None
            
        new_hash = get_content_hash(new_content)
        
        # Get list of all files in upload directory
        existing_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) 
                         if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and allowed_file(f)]
        
        # Check if this exact file already exists
        for existing_file in existing_files:
            # Skip comparing with itself
            if os.path.samefile(new_file_path, existing_file):
                continue
                
            existing_content = extract_file_content(existing_file)
            existing_hash = get_content_hash(existing_content)
            
            # First check: exact hash match (faster)
            if new_hash == existing_hash:
                # Get collection name from file
                collection_name = os.path.splitext(os.path.basename(existing_file))[0].replace(" ", "_")
                return True, collection_name
            
            # Second check: content similarity for partial matches
            # Only do this check if content is not too large to avoid performance issues
            if len(new_content) < 100000 and len(existing_content) < 100000:
                similarity = SequenceMatcher(None, new_content, existing_content).ratio()
                if similarity >= similarity_threshold:
                    # Get collection name from file
                    collection_name = os.path.splitext(os.path.basename(existing_file))[0].replace(" ", "_")
                    return True, collection_name
        
        # No similar file found, use the new file's name for collection
        collection_name = os.path.splitext(os.path.basename(new_file_path))[0].replace(" ", "_")
        return False, collection_name
    except Exception as e:
        print(f"Error checking for similar files: {str(e)}")
        # Default to using the filename as collection name
        collection_name = os.path.splitext(os.path.basename(new_file_path))[0].replace(" ", "_")
        return False, collection_name


def process_file_for_rag(file_path, model_embed="nomic-embed-text:latest"):
    """
    Process a file and create a vector store for RAG
    
    Args:
        file_path (str): Path to the uploaded file
        model_embed (str): Ollama embedding model to use
    
    Returns:
        tuple: (collection_name, is_new_collection) - Name of the collection and whether it's new
    """
    try:
        # Check if file extension is allowed
        if not allowed_file(os.path.basename(file_path)):
            raise ValueError(f"Unsupported file extension: {get_file_extension(file_path)}")
        
        # Check if similar content already exists and get the collection name to use
        is_duplicate, collection_name = find_similar_file(file_path)
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Check if collection already exists in ChromaDB
        collection_exists = False
        try:
            collection = client.get_collection(name=collection_name)
            collection_exists = True
            doc_count = collection.count()
            print(f"Found existing collection '{collection_name}' with {doc_count} documents")
            
            # If the collection exists and has documents, use it as is
            if doc_count > 0:
                print(f"Using existing collection '{collection_name}' without re-embedding")
                return collection_name, False
                
        except Exception as e:
            collection_exists = False
            print(f"Collection '{collection_name}' does not exist in ChromaDB: {str(e)}")
        
        # Process the document and create/update the collection
        print(f"Processing document for collection '{collection_name}'...")
        file_extension = get_file_extension(file_path)
        documents = process_document(file_path, file_extension)
        
        # Always set recreate=False to avoid deleting existing data
        create_vector_store(documents, model_embed, os.path.basename(file_path), recreate=not collection_exists)
        
        return collection_name, True
        
    except Exception as e:
        print(f"Error processing file for RAG: {str(e)}")
        traceback.print_exc()
        raise


async def query_with_hybrid_rag(query, collection_name, n_results=5, model_embed="nomic-embed-text:latest", model_llm="llama2:13b"):
    """
    Retrieve relevant content and generate an augmented response using hybrid search
    
    Args:
        query: User query
        collection_name: Name of the collection to query
        n_results: Number of results to retrieve
        model_embed: Embedding model to use
        model_llm: LLM model to use
        
    Returns:
        dict: Response with context, insights and generated answer
    """
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Get the collection
        try:
            collection = client.get_collection(name=collection_name)
        except Exception as e:
            return {
                "error": f"Collection not found: {collection_name}",
                "details": str(e)
            }
        
        # Initialize keyword index with documents from collection
        print("Creating keyword index from documents...")
        keyword_index = KeywordIndex()
        
        # Get all documents from the collection for indexing
        collection_data = collection.get(include=["documents", "embeddings", "metadatas"])
        documents = []
        
        for i, doc in enumerate(collection_data["documents"]):
            documents.append({
                "id": collection_data["ids"][i],
                "content": doc
            })
        
        keyword_index.index_documents(documents)
        
        # Create hybrid retrieval orchestrator
        orchestrator = HybridRetrievalOrchestrator(
            vector_store=collection,
            keyword_index=keyword_index
        )
        orchestrator.set_embed_model(model_embed)
        
        # Create reasoning module
        reasoning_module = HybridReasoningModule(llm_model=model_llm)
        
        # Retrieve documents using hybrid search
        print(f"Retrieving documents for query: {query}")
        hybrid_results = await orchestrator.retrieve(query, top_k=n_results)
        
        # Process and analyze context
        print("Analyzing search results...")
        analysis = await reasoning_module.process_context(hybrid_results)
        
        # Prepare context for LLM
        context = prepare_context_from_hybrid_results(hybrid_results)
        
        # Create the QA chain
        print(f"Creating QA chain with {model_llm}")
        qa_chain = create_qa_chain(model_name=model_llm)
        
        # Run the chain
        print(f"Generating response")
        response = qa_chain.invoke({
            "context": context,
            "question": query
        })
        
        # Return the results
        return {
            "query": query,
            "context": [doc.get('document', {}).get('content', '') for doc in hybrid_results],
            "response": response,
            "insights": analysis.get("insights", []),
            "hybrid_info": {
                "methods_used": list(set(sum([doc.get('methods', []) for doc in hybrid_results], []))),
                "result_count": len(hybrid_results)
            }
        }
        
    except Exception as e:
        print(f"Error during hybrid RAG process: {str(e)}")
        traceback.print_exc()
        return {
            "error": "Error during hybrid RAG process",
            "details": str(e)
        }


def create_qa_chain(model_name="llama2:13b"):
    """
    Create a simple question-answering chain using LangChain
    
    Args:
        model_name (str): Name of the Ollama model to use
    
    Returns:
        chain: A LangChain chain for QA
    """
    # Initialize the LLM
    llm = ChatOllama(model=model_name)
    
    # Create the prompt template
    template = """
    Answer the question based on the context below. If you can't
    answer the question based on the provided context, reply "I don't know".
    
    Context: {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the output parser
    parser = StrOutputParser()
    
    # Create the chain
    chain = prompt | llm | parser
    
    return chain


# Function to run async functions from sync code
def run_async(async_func, *args, **kwargs):
    """Helper function to run async functions"""
    return asyncio.run(async_func(*args, **kwargs))


# Synchronous wrapper for async hybrid RAG query function
def query_with_rag(query, collection_name, n_results=5, model_embed="nomic-embed-text:latest", model_llm="llama2:13b"):
    """
    Synchronous wrapper for async hybrid RAG query function
    
    Args:
        query: User query
        collection_name: Name of the collection to query
        n_results: Number of results to retrieve
        model_embed: Embedding model to use
        model_llm: LLM model to use
        
    Returns:
        dict: Response with context, insights and generated answer
    """
    return run_async(query_with_hybrid_rag, query, collection_name, n_results, model_embed, model_llm)


# Create necessary directories
for directory in [UPLOAD_DIR, CHROMA_DIR]:
    os.makedirs(directory, exist_ok=True)
"""
Enhanced Vector Store Manager with Persistent Storage
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Advanced vector store manager with persistence and optimization"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store manager
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            embedding_model: HuggingFace embedding model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings(embedding_model)
        
        logger.info(f"VectorStoreManager initialized: {collection_name}")
    
    def _initialize_embeddings(self, model_name: str):
        """Initialize embedding model with error handling"""
        logger.info(f"Loading embedding model: {model_name}")
        logger.info("(This may take a moment on first run...)")
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            # Test embeddings
            test_embedding = embeddings.embed_query("test")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding model returned empty embeddings")
            
            logger.info(f"✅ Embedding model loaded (dimension: {len(test_embedding)})")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {str(e)}")
            raise
    
    def create_vectorstore(self, chunks: List, batch_size: int = 100):
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks
            batch_size: Batch size for processing
            
        Returns:
            Chroma vectorstore instance (not wrapped)
        """
        if not chunks:
            raise ValueError("No document chunks provided")
        
        logger.info(f"\n🔢 Creating vector store from {len(chunks)} chunks...")
        
        # Validate chunks
        valid_chunks = [c for c in chunks if c.page_content.strip()]
        if not valid_chunks:
            raise ValueError("All document chunks are empty")
        
        logger.info(f"Valid chunks: {len(valid_chunks)}")
        
        # Clear existing vectorstore if it exists
        self._clear_existing_store()
        
        try:
            logger.info("Generating embeddings (this may take 1-3 minutes)...")
            
            # Process in batches to avoid memory issues
            if len(valid_chunks) <= batch_size:
                # Single batch
                vectorstore = Chroma.from_documents(
                    documents=valid_chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                logger.info(f"✅ Created vectorstore with {len(valid_chunks)} chunks")
            else:
                # Multiple batches
                # First batch
                first_batch = valid_chunks[:batch_size]
                vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
                logger.info(f"✅ Created vectorstore with first {len(first_batch)} chunks")
                
                # Remaining batches
                for i in range(batch_size, len(valid_chunks), batch_size):
                    batch = valid_chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                    progress = min(i + batch_size, len(valid_chunks))
                    logger.info(f"Progress: {progress}/{len(valid_chunks)} chunks processed")
            
            # Verify creation
            total_count = vectorstore._collection.count()
            logger.info(f"✅ Vector store created successfully with {total_count} vectors")
            
            # Store reference
            self.vectorstore = vectorstore
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Optional["Chroma"]:
        """Load existing vector store from disk"""
        if not self._store_exists():
            logger.warning(f"Vector store not found at {self.persist_directory}")
            return None
        
        try:
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            count = self.vectorstore._collection.count()
            logger.info(f"✅ Loaded vector store with {count} vectors")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {str(e)}")
            return None
    
    def _store_exists(self) -> bool:
        """Check if vector store exists on disk"""
        return os.path.exists(self.persist_directory) and \
               os.path.isdir(self.persist_directory) and \
               len(os.listdir(self.persist_directory)) > 0
    
    def _clear_existing_store(self):
        """Clear existing vector store"""
        if self._store_exists():
            try:
                logger.info("Clearing existing vector store...")
                
                # Close existing connection
                if self.vectorstore is not None:
                    try:
                        del self.vectorstore
                    except:
                        pass
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Remove directory
                shutil.rmtree(self.persist_directory)
                logger.info("✅ Existing vector store cleared")
                
                # Recreate directory
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not fully clear existing store: {str(e)}")
    
    def add_documents(self, chunks: List):
        """Add documents to existing vectorstore"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        logger.info(f"Adding {len(chunks)} chunks to vectorstore...")
        self.vectorstore.add_documents(chunks)
        logger.info("✅ Documents added")
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        """Perform similarity search"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List:
        """Perform similarity search with relevance scores"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """Get retriever for chain integration"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def delete_vectorstore(self):
        """Completely delete the vector store"""
        logger.info("Deleting vector store...")
        
        # Close connection
        if self.vectorstore is not None:
            try:
                del self.vectorstore
                self.vectorstore = None
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Delete directory
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("✅ Vector store deleted")
            except Exception as e:
                logger.error(f"❌ Error deleting vector store: {str(e)}")
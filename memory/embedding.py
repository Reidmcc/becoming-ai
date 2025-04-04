from typing import List, Optional, Union, Dict, Any
import logging
import numpy as np

class EmbeddingService:
    """Abstract base class for embedding services"""
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Convert text to vector embedding
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        raise NotImplementedError("Subclasses must implement get_embedding")


class FrontierEmbeddingService(EmbeddingService):
    """Embedding service using the Frontier API"""
    
    def __init__(self, frontier_client):
        """
        Initialize with a frontier client
        
        Args:
            frontier_client: Instance of FrontierClient to use for embeddings
        """
        self.client = frontier_client
        self.logger = logging.getLogger("FrontierEmbeddingService")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding via the Frontier client
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        try:
            # Trim text if too long (most embedding APIs have limits)
            trimmed_text = text[:8000] if len(text) > 8000 else text
            
            # Use the client's embedding method
            embedding = self.client.get_embedding(trimmed_text)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector of appropriate dimension as fallback
            # TODO set this value when initializing based on specific service
            return [0.0] * 1536


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Local embedding service using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize with a model name
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.logger = logging.getLogger("SentenceTransformerEmbeddingService")
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding using sentence-transformers
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        try:
            # Truncate if text is too long (to avoid OOM errors)
            # Most models work well with texts up to ~512 tokens
            trimmed_text = text[:2000] if len(text) > 2000 else text
            
            # Get embedding
            embedding = self.model.encode(trimmed_text)
            
            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            # TODO set this value when initializing based on specific model
            return [0.0] * 768  # Common dimension for sentence-transformers
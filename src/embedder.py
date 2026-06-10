from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union


class EmbeddingEngine:
    """Embedding generation using pre-trained Sentence Transformer models."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding engine with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained Sentence Transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a dense vector embedding for the given text.
        
        Args:
            text: Input text string to embed
            
        Returns:
            Numpy array representing the text embedding
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Generate embedding
        embedding = self.model.encode(text, show_progress_bar=False)
        
        return embedding
    
    def generate_embeddings_batch(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        # Generate embeddings in batch
        embeddings = self.model.encode(valid_texts, show_progress_bar=False)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        # Generate a dummy embedding to get the dimension
        dummy_embedding = self.generate_embedding("test")
        return dummy_embedding.shape[0]


if __name__ == "__main__":
    # Example usage
    embedder = EmbeddingEngine()
    
    test_text = "Machine learning engineer with experience in Python and TensorFlow."
    embedding = embedder.generate_embedding(test_text)
    
    print(f"Model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

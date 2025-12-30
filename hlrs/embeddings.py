"""
Embeddings module using Sentence Transformers and FAISS for fast retrieval
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from rich.console import Console

console = Console()


class EmbeddingEngine:
    """Handles embeddings and vector similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        console.print(f"[cyan]Loading embedding model: {model_name}...[/cyan]")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.llr_ids = []
        console.print(f"[green]✓ Model loaded (dimension: {self.dimension})[/green]")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of embeddings
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def build_llr_index(self, llrs: List[Dict]):
        """
        Build FAISS index for LLRs
        
        Args:
            llrs: List of LLR dictionaries with 'id' and 'embedding_text'
        """
        console.print("[cyan]Building FAISS index for LLRs...[/cyan]")
        
        # Extract texts and IDs
        self.llr_ids = [llr['id'] for llr in llrs]
        llr_texts = [llr['embedding_text'] for llr in llrs]
        
        # Generate embeddings
        embeddings = self.encode(llr_texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index (IndexFlatIP for inner product = cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        console.print(f"[green]✓ FAISS index built with {len(llrs)} LLRs[/green]")
    
    def search_top_k(self, query_text: str, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search for top-K similar LLRs using FAISS
        
        Args:
            query_text: HLR text to search for
            k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (llr_id, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_llr_index first.")
        
        # Encode query
        query_embedding = self.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if sim >= threshold:
                results.append((self.llr_ids[idx], float(sim)))
        
        return results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.encode([text1, text2])
        faiss.normalize_L2(embeddings)
        
        # Cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

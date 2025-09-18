import openai
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, api_key: str, base_url: str, embedding_model: str, embedding_dim: int = 768):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dim

    def retrieve_memories(self, query: str, memories: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on embedding similarity"""
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Get embeddings for all memories
        memory_embeddings = []
        for memory in memories:
            # Use memory content for embedding
            content = memory["content"]
            embedding = self.get_embedding(content)
            memory_embeddings.append(embedding)

        # Calculate similarity scores
        similarities = cosine_similarity(
            [query_embedding], memory_embeddings)[0]

        # Add similarity scores to memories and sort
        memories_with_scores = []
        for i, memory in enumerate(memories):
            memories_with_scores.append({
                **memory,
                "relevance_score": float(similarities[i])
            })

        # Sort by relevance score (descending)
        memories_with_scores.sort(
            key=lambda x: x["relevance_score"], reverse=True)

        # Return top-k memories
        return memories_with_scores[:top_k]

    def retrieve_memories_hybrid(self, query: str, memories: List[Dict[str, Any]], top_k: int = 5,
                                 embedding_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining embedding similarity and keyword matching"""
        # Get embedding-based scores
        query_embedding = self.get_embedding(query)
        memory_embeddings = []

        for memory in memories:
            content = memory["content"]
            embedding = self.get_embedding(content)
            memory_embeddings.append(embedding)

        embedding_similarities = cosine_similarity(
            [query_embedding], memory_embeddings)[0]

        # Get keyword-based scores
        query_lower = query.lower()
        keyword_scores = []

        for memory in memories:
            content_lower = memory["content"].lower()
            # Simple keyword matching: count of query words in memory
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content_lower)
            keyword_score = matches / len(query_words) if query_words else 0
            keyword_scores.append(keyword_score)

        # Normalize scores
        if np.max(embedding_similarities) > 0:
            embedding_similarities = embedding_similarities / \
                np.max(embedding_similarities)

        if np.max(keyword_scores) > 0:
            keyword_scores = keyword_scores / np.max(keyword_scores)

        # Combine scores
        combined_scores = embedding_weight * embedding_similarities + \
            (1 - embedding_weight) * np.array(keyword_scores)

        # Add combined scores to memories and sort
        memories_with_scores = []
        for i, memory in enumerate(memories):
            memories_with_scores.append({
                **memory,
                "relevance_score": float(combined_scores[i]),
                "embedding_score": float(embedding_similarities[i]),
                "keyword_score": float(keyword_scores[i])
            })

        # Sort by combined relevance score (descending)
        memories_with_scores.sort(
            key=lambda x: x["relevance_score"], reverse=True)

        # Return top-k memories
        return memories_with_scores[:top_k]

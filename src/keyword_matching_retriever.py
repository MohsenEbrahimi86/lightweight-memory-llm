from typing import List, Dict, Any, Tuple


class KeywordMatchingRetriever:
    def retrieve_memories(
        self, query: str, memories: List[Dict[str, Any]], top_k: int = 5
    ):
        query_lower = query.lower()
        results = []

        for memory in memories:
            content = memory["content"].lower()
            # Simple keyword matching
            matches = sum(1 for word in query_lower.split() if word in content)
            relevance = matches / len(query_lower.split()) if query_lower.split() else 0

            results.append({**memory, "relevance_score": relevance})

        # Sort by relevance and return top-k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

import os
import yaml
from src.extractor import MemoryExtractor
from src.store import MemoryStore
from src.retriever import Retriever
from typing import List, Dict, Any, Tuple

class MemorySystem:
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.extractor = MemoryExtractor(
            api_key=self.config["api"]["api_key"],
            base_url=self.config["api"]["base_url"],
            model=self.config["api"]["chat_model"]
        )

        self.store = MemoryStore(
            store_path=self.config["memory"]["store_path"])

        self.retriever = Retriever(
            api_key=self.config["api"]["api_key"],
            base_url=self.config["api"]["base_url"],
            embedding_model=self.config["api"]["embedding_model"],
            embedding_dim=self.config["memory"]["embedding_dim"]
        )

    def process_conversation(self, conversation: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a conversation and update memories"""
        # Get existing memories
        existing_memories = self.store.get_all_memories()

        # Extract new facts
        new_memories = self.extractor.extract_facts(
            conversation, existing_memories)

        # Add to store
        for memory in new_memories:
            self.store.add_memory(memory)

        return new_memories

    def query_memories(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Query memories based on a question"""
        if top_k is None:
            top_k = self.config["memory"]["max_memories_retrieved"]

        # Get all memories
        all_memories = self.store.get_all_memories()

        # Retrieve relevant memories
        relevant_memories = self.retriever.retrieve_memories(
            query, all_memories, top_k)

        return relevant_memories

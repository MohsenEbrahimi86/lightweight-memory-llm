import orjson as json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class SimpleJsonStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.memories = []
        self._load_store()

    def _load_store(self):
        """Load memories from file if it exists"""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r') as f:
                    self.memories = json.load(f)
            except Exception as e:
                print(f"Error loading memory store: {e}")
                self.memories = []

    def _save_store(self):
        """Save memories to file"""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def add_memory(self, memory: Dict[str, Any]) -> str:
        """
        Add a new memory to the store
        """
        # Check if this is an update to an existing memory
        if memory.get("is_update", False) and "previous_value" in memory:
            # Find the memory being updated
            for i, existing_memory in enumerate(self.memories):
                if existing_memory["content"] == memory["previous_value"]:
                    # Update the existing memory
                    self.memories[i] = {
                        **existing_memory,
                        "content": memory["content"],
                        "confidence": memory["confidence"],
                        "timestamp": memory["timestamp"],
                        "previous_value": memory["previous_value"],
                        "updated_from": memory["extracted_from"]
                    }
                    self._save_store()
                    return self.memories[i]["fact_id"]

        # If not an update or no matching memory found, add as new
        self.memories.append(memory)
        self._save_store()
        return memory["fact_id"]

    def get_memory(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        for memory in self.memories:
            if memory["fact_id"] == fact_id:
                return memory
        return None

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories in the store"""
        return self.memories

    def update_memory(self, fact_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory"""
        for i, memory in enumerate(self.memories):
            if memory["fact_id"] == fact_id:
                self.memories[i] = {**memory, **updates}
                self.memories[i]["timestamp"] = datetime.now().isoformat()
                self._save_store()
                return True
        return False

    def delete_memory(self, fact_id: str) -> bool:
        """Delete a memory from the store"""
        for i, memory in enumerate(self.memories):
            if memory["fact_id"] == fact_id:
                del self.memories[i]
                self._save_store()
                return True
        return False

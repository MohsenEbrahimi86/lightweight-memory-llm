import os
import yaml
from src.extractor import MemoryExtractor
from src.store import MemoryStore
from src.retriever import Retriever
from benchmark.generate_data import ConversationGenerator
from benchmark.evaluate import Evaluator


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


def main():
    # Initialize memory system
    memory_system = MemorySystem()

    # Initialize conversation generator
    generator = ConversationGenerator(
        api_key=memory_system.config["api"]["api_key"],
        base_url=memory_system.config["api"]["base_url"],
        model=memory_system.config["api"]["chat_model"]
    )

    # Initialize evaluator
    evaluator = Evaluator()

    # Generate synthetic conversations
    print("Generating synthetic conversations...")
    conversations = generator.generate_conversations(num_conversations=50)

    # Generate expected memories
    print("Generating expected memories...")
    expected_memories = generator.generate_expected_memories(conversations)

    # Process conversations with memory system
    print("Processing conversations with memory system...")
    for i, conversation in enumerate(conversations):
        print(f"Processing conversation {i+1}/{len(conversations)}...")
        memory_system.process_conversation(conversation["turns"])

    # Evaluate memory system
    print("Evaluating memory system...")
    evaluation_results = evaluator.run_full_evaluation(
        memory_system, conversations, expected_memories)

    # Compare with baseline
    print("Comparing with baseline...")
    comparison_results = evaluator.compare_with_baseline(
        memory_system, conversations, expected_memories)

    print("Evaluation complete!")
    print(
        f"Results saved to {memory_system.config['benchmark']['results_path']}")


if __name__ == "__main__":
    main()

import os
import yaml
from src.extractor import MemoryExtractor
from src.store import SimpleJsonStore
from src.retriever import Retriever
from benchmark.generate_data import ConversationGenerator
from benchmark.evaluate import Evaluator
from typing import List, Dict, Any, Tuple
from memory_system import MemorySystem


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
        print(f"Processing conversation {i + 1}/{len(conversations)}...")
        memory_system.process_conversation(conversation["turns"])

    # Evaluate memory system
    print("Evaluating memory system...")
    evaluation_results = evaluator.run_full_evaluation(
        memory_system,
        conversations,
        expected_memories
    )

    # Compare with baseline
    print("Comparing with baseline...")
    comparison_results = evaluator.compare_with_baseline(
        memory_system, conversations, expected_memories)

    print("Evaluation complete!")
    print(
        f"Results saved to {memory_system.config['benchmark']['results_path']}")


if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_recall_fscore_support
import os

from src.keyword_matching_retriever import KeywordMatchingRetriever


class Evaluator:
    def __init__(self, results_path: str = "benchmark/results/"):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)

    def evaluate_extraction_quality(
        self,
        expected_memories: List[Dict[str, Any]],
        extracted_memories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate the quality of memory extraction"""
        # Group memories by conversation
        expected_by_conv = {}
        extracted_by_conv = {}

        for memory in expected_memories:
            conv_idx = memory["conversation_index"]
            if conv_idx not in expected_by_conv:
                expected_by_conv[conv_idx] = []
            expected_by_conv[conv_idx].append(memory)

        for memory in extracted_memories:
            conv_idx = memory.get("conversation_index", -1)
            if conv_idx not in extracted_by_conv:
                extracted_by_conv[conv_idx] = []
            extracted_by_conv[conv_idx].append(memory)

        # Calculate metrics for each conversation
        all_precisions = []
        all_recalls = []
        all_f1s = []

        for conv_idx in expected_by_conv:
            expected = expected_by_conv[conv_idx]
            extracted = extracted_by_conv.get(conv_idx, [])

            # Simple matching based on content similarity
            y_true = [1] * len(expected)  # All expected memories are true
            y_pred = [0] * len(expected)  # Initialize as not matched

            # Try to match extracted memories to expected ones
            for i, exp_mem in enumerate(expected):
                for ext_mem in extracted:
                    # Simple check if content is similar
                    if self._is_similar_content(exp_mem["content"], ext_mem["content"]):
                        y_pred[i] = 1
                        break

            # Calculate precision, recall, F1
            if len(extracted) > 0:
                precision = sum(y_pred) / len(extracted)
            else:
                precision = 0

            recall = sum(y_pred) / len(expected)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)

        # Calculate average metrics
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1s)

        return {"precision": avg_precision, "recall": avg_recall, "f1_score": avg_f1}

    def evaluate_retrieval_precision(
        self,
        queries: List[str],
        retrieved_memories: List[List[Dict[str, Any]]],
        relevant_memories: List[List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """Evaluate the precision of memory retrieval"""
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_ndcgs = []

        for i, (query, retrieved, relevant) in enumerate(
            zip(queries, retrieved_memories, relevant_memories)
        ):
            # Convert to binary relevance (1 if relevant, 0 otherwise)
            y_true = [1] * len(relevant)
            y_pred = [0] * len(relevant)

            # Check which retrieved memories are relevant
            for j, ret_mem in enumerate(retrieved):
                for rel_mem in relevant:
                    if self._is_similar_content(ret_mem["content"], rel_mem["content"]):
                        y_pred[min(j, len(y_pred) - 1)] = 1
                        break

            # Calculate precision at k
            precision_at_k = sum(y_pred) / len(retrieved) if retrieved else 0

            # Calculate recall
            recall = sum(y_pred) / len(relevant) if relevant else 0

            # Calculate F1
            if precision_at_k + recall > 0:
                f1 = 2 * precision_at_k * recall / (precision_at_k + recall)
            else:
                f1 = 0

            # Calculate NDCG (simplified version)
            ndcg = self._calculate_ndcg(retrieved, relevant)

            all_precisions.append(precision_at_k)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_ndcgs.append(ndcg)

        # Calculate average metrics
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1s)
        avg_ndcg = np.mean(all_ndcgs)

        return {
            "precision_at_k": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "ndcg": avg_ndcg,
        }

    def evaluate_update_accuracy(
        self,
        original_memories: List[Dict[str, Any]],
        updated_memories: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate the accuracy of memory updates"""
        correct_updates = 0
        total_updates = len(updated_memories)

        for updated_mem in updated_memories:
            # Find the corresponding original memory
            original_found = False
            for original_mem in original_memories:
                if self._is_similar_content(
                    original_mem["content"], updated_mem.get("previous_value", "")
                ):
                    original_found = True
                    break

            if original_found:
                correct_updates += 1

        accuracy = correct_updates / total_updates if total_updates > 0 else 0

        return {
            "update_accuracy": accuracy,
            "correct_updates": correct_updates,
            "total_updates": total_updates,
        }

    def evaluate_memory_consistency(
        self, memories: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate the consistency of memories (custom metric)"""
        # Check for duplicate or contradictory memories
        contradictions = 0
        duplicates = 0

        for i, mem1 in enumerate(memories):
            for j, mem2 in enumerate(memories[i + 1 :], i + 1):
                # Check for duplicates (very similar content)
                if self._is_similar_content(
                    mem1["content"], mem2["content"], threshold=0.9
                ):
                    duplicates += 1

                # Check for contradictions (e.g., "works at X" vs "works at Y")
                if self._is_contradictory(mem1["content"], mem2["content"]):
                    contradictions += 1

        # Calculate consistency score (1 - normalized contradictions and duplicates)
        total_pairs = len(memories) * (len(memories) - 1) / 2
        consistency_score = (
            1 - (contradictions + duplicates) / total_pairs if total_pairs > 0 else 1
        )

        return {
            "consistency_score": consistency_score,
            "contradictions": contradictions,
            "duplicates": duplicates,
            "total_memory_pairs": total_pairs,
        }

    def run_full_evaluation(
        self,
        memory_system,
        conversations: List[Dict[str, Any]],
        expected_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a full evaluation of the memory system"""
        # Test extraction
        all_extracted_memories = []
        for i, conversation in enumerate(conversations):
            extracted = memory_system.extractor.extract_facts(conversation["turns"])

            for mem in extracted:
                mem["conversation_index"] = i
            all_extracted_memories.extend(extracted)

        # Evaluate extraction quality
        extraction_metrics = self.evaluate_extraction_quality(
            expected_memories, all_extracted_memories
        )

        # Test retrieval
        queries = []
        retrieved_memories_list = []
        relevant_memories_list = []

        for i, conversation in enumerate(conversations):
            # Generate a query based on the conversation
            if conversation["turns"]:
                # Use the last user turn as a query
                for turn in reversed(conversation["turns"]):
                    if turn["role"] == "user":
                        query = turn["content"]
                        break
                else:
                    query = "Tell me about the user"
            else:
                query = "Tell me about the user"

            queries.append(query)

            # Get relevant memories for this conversation
            relevant = [
                mem for mem in expected_memories if mem["conversation_index"] == i
            ]
            relevant_memories_list.append(relevant)

            # Retrieve memories using the system
            all_stored_memories = memory_system.store.get_all_memories()
            retrieved = memory_system.retriever.retrieve_memories(
                query, all_stored_memories
            )
            retrieved_memories_list.append(retrieved)

        # Evaluate retrieval precision
        retrieval_metrics = self.evaluate_retrieval_precision(
            queries, retrieved_memories_list, relevant_memories_list
        )

        # Test updates
        # Create some test conversations with updates
        update_conversations = [
            {
                "turns": [
                    {"role": "user", "content": "I work at Google."},
                    {
                        "role": "assistant",
                        "content": "That's interesting! What do you do at Google?",
                    },
                    {
                        "role": "user",
                        "content": "Actually, I just started working at Microsoft.",
                    },
                ]
            },
            {
                "turns": [
                    {"role": "user", "content": "I live in New York."},
                    {
                        "role": "assistant",
                        "content": "How do you like living in New York?",
                    },
                    {"role": "user", "content": "I recently moved to Boston."},
                ]
            },
        ]

        original_memories = []
        updated_memories = []

        for conversation in update_conversations:
            # Extract original facts
            original = memory_system.extractor.extract_facts(conversation["turns"][:2])
            original_memories.extend(original)

            # Extract updated facts
            updated = memory_system.extractor.extract_facts(conversation["turns"])
            updated_memories.extend(updated)

        # Evaluate update accuracy
        update_metrics = self.evaluate_update_accuracy(
            original_memories, updated_memories
        )

        # Evaluate memory consistency
        all_memories = memory_system.store.get_all_memories()
        consistency_metrics = self.evaluate_memory_consistency(all_memories)

        # Combine all metrics
        all_metrics = {
            "extraction_quality": extraction_metrics,
            "retrieval_precision": retrieval_metrics,
            "update_accuracy": update_metrics,
            "memory_consistency": consistency_metrics,
        }

        # Save results
        with open(f"{self.results_path}evaluation_results.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Create visualizations
        self._create_visualizations(all_metrics)

        return all_metrics

    def compare_with_baseline(
        self,
        memory_system,
        conversations: List[Dict[str, Any]],
        expected_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare the memory system with a baseline (simple keyword matching)"""
        # Evaluate our system
        system_results = self.run_full_evaluation(
            memory_system, conversations, expected_memories
        )

        # Create a simple baseline (keyword matching)
        # Replace the retriever with baseline
        original_retriever = memory_system.retriever
        memory_system.retriever = KeywordMatchingRetriever()

        # Evaluate baseline
        baseline_results = self.run_full_evaluation(
            memory_system, conversations, expected_memories
        )

        # Restore original retriever
        memory_system.retriever = original_retriever

        # Compare results
        comparison = {}
        for metric in system_results:
            comparison[metric] = {
                "system": system_results[metric],
                "baseline": baseline_results[metric],
                "improvement": {},
            }

            for submetric in system_results[metric]:
                sys_val = system_results[metric][submetric]
                base_val = baseline_results[metric][submetric]
                improvement = (sys_val - base_val) / base_val if base_val != 0 else 0

                comparison[metric]["improvement"][submetric] = {
                    "absolute": sys_val - base_val,
                    "percent": improvement * 100,
                }

        # Save comparison results
        with open(f"{self.results_path}baseline_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # Create comparison visualizations
        self._create_comparison_visualizations(comparison)

        return comparison

    def _is_similar_content(
        self, content1: str, content2: str, threshold: float = 0.7
    ) -> bool:
        """Check if two content strings are similar"""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard = len(intersection) / len(union)
        return jaccard >= threshold

    def _is_contradictory(self, content1: str, content2: str) -> bool:
        """Check if two content strings are contradictory"""
        # Simple check for common contradictions
        contradictions = [
            ("work at", "work at"),
            ("live in", "live in"),
            ("name is", "name is"),
        ]

        for trigger1, trigger2 in contradictions:
            if trigger1 in content1.lower() and trigger2 in content2.lower():
                # Extract the values after the trigger
                val1 = (
                    content1.lower().split(trigger1)[1].strip().split()[0]
                    if trigger1 in content1.lower()
                    else ""
                )
                val2 = (
                    content2.lower().split(trigger2)[1].strip().split()[0]
                    if trigger2 in content2.lower()
                    else ""
                )

                if val1 and val2 and val1 != val2:
                    return True

        return False

    def _calculate_ndcg(
        self, retrieved: List[Dict[str, Any]], relevant: List[Dict[str, Any]]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (simplified version)"""
        # Create relevance scores (1 if relevant, 0 otherwise)
        relevance_scores = []

        for i, ret_mem in enumerate(retrieved):
            is_relevant = 0
            for rel_mem in relevant:
                if self._is_similar_content(ret_mem["content"], rel_mem["content"]):
                    is_relevant = 1
                    break
            relevance_scores.append(is_relevant)

        # Calculate DCG
        dcg = 0
        for i, score in enumerate(relevance_scores):
            dcg += score / (i + 1)  # Simplified: log2(i+2) -> i+1

        # Calculate IDCG (perfect ranking)
        idcg = sum([1 / (i + 1) for i in range(len(relevant))])

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        return ndcg

    def _create_visualizations(self, metrics: Dict[str, Any]):
        """Create visualizations for evaluation metrics"""
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Memory System Evaluation Results", fontsize=16)

        # Plot extraction quality
        extraction_metrics = metrics["extraction_quality"]
        axs[0, 0].bar(extraction_metrics.keys(), extraction_metrics.values())
        axs[0, 0].set_title("Extraction Quality")
        axs[0, 0].set_ylim(0, 1)

        # Plot retrieval precision
        retrieval_metrics = metrics["retrieval_precision"]
        axs[0, 1].bar(retrieval_metrics.keys(), retrieval_metrics.values())
        axs[0, 1].set_title("Retrieval Precision")
        axs[0, 1].set_ylim(0, 1)

        # Plot update accuracy
        update_metrics = metrics["update_accuracy"]
        axs[1, 0].bar(update_metrics.keys(), [update_metrics["update_accuracy"]])
        axs[1, 0].set_title("Update Accuracy")
        axs[1, 0].set_ylim(0, 1)

        # Plot memory consistency
        consistency_metrics = metrics["memory_consistency"]
        axs[1, 1].bar(
            consistency_metrics.keys(), [consistency_metrics["consistency_score"]]
        )
        axs[1, 1].set_title("Memory Consistency")
        axs[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{self.results_path}evaluation_metrics.png")
        plt.close()

    def _create_comparison_visualizations(self, comparison: Dict[str, Any]):
        """Create visualizations for baseline comparison"""
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Memory System vs Baseline Comparison", fontsize=16)

        # Flatten the metrics for easier plotting
        metrics_to_plot = [
            ("extraction_quality", "f1_score"),
            ("retrieval_precision", "precision_at_k"),
            ("update_accuracy", "update_accuracy"),
            ("memory_consistency", "consistency_score"),
        ]

        for i, (metric, submetric) in enumerate(metrics_to_plot):
            row = i // 2
            col = i % 2

            system_value = comparison[metric]["system"][submetric]
            baseline_value = comparison[metric]["baseline"][submetric]

            axs[row, col].bar(["System", "Baseline"], [system_value, baseline_value])
            axs[row, col].set_title(
                f"{metric.replace('_', ' ').title()} - {submetric.replace('_', ' ').title()}"
            )
            axs[row, col].set_ylim(0, 1)

            # Add improvement percentage
            improvement = comparison[metric]["improvement"][submetric]["percent"]
            axs[row, col].text(
                0.5,
                0.9,
                f"{improvement:.1f}%",
                transform=axs[row, col].transAxes,
                ha="center",
                va="top",
                color="green" if improvement > 0 else "red",
            )

        plt.tight_layout()
        plt.savefig(f"{self.results_path}baseline_comparison.png")
        plt.close()

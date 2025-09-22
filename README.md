# lightweight-memory-llm

A lightweight memory layer for LLMs and a comprehensive benchmark to evaluate its performance.

## Overview

This project implements a lightweight memory layer for large language models (LLMs) and a comprehensive benchmark to evaluate its performance. The system extracts key facts from conversations, stores them with metadata, and retrieves relevant memories based on queries.

## System Architecture

The memory system consists of three main components:

1. **Memory Extractor**: Uses a transformer model to extract key facts from conversation turns, handling both new facts and updates to existing facts.

2. **Memory Store**: Stores memories with metadata (timestamps, confidence scores) and supports write, read, and update operations.

3. **Retriever**: Uses embedding-based similarity search to return relevant memories for a given query.

## Project Structure
```
repo/
├── README.md
├── main.py
├── src/
│ ├── init.py
│ ├── extractor.py
│ ├── store.py
│ └── retriever.py
├── benchmark/
│ ├── init.py
│ ├── generate_data.py
│ ├── evaluate.py
│ └── results/
├── configs/
│ └── config.yaml
├── demo/
│ └── demo_notebook.ipynb
├── data/
└── requirements.txt
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- API key for the LiteLLM proxy

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the API key in configs/config.yaml:

```yaml
api:
  api_key: "YOUR_API_KEY_HERE"
```

### Running the System

1. Generate synthetic conversations and evaluate the system:

```bash
python main.py
```

2. Run the demo notebook:

```bash
jupyter notebook demo/demo_notebook.ipynb
```

### Benchmark Design

The benchmark includes:

- **Synthetic Dataset**: 50 multi-turn conversations covering three scenarios:

  - Simple fact storage and retrieval
  - Fact updates and corrections
  - Multi-hop reasoning

- **Evaluation Metrics**:

  - **Extraction quality**: Measures if correct facts are captured
  - **Retrieval precision**: Measures if retrieved memories are relevant
  - **Update accuracy**: Measures if updates work correctly
  - **Memory consistency**: Measures if memories are consistent and non-contradictory

- **Baseline Comparison**: Compares the system against a simple keyword matching baseline

## Results

The evaluation results are saved in `benchmark/results/` and include:

- `evaluation_results.json`: Metrics for the memory system
- `baseline_comparison.json`: Comparison with baseline
- Visualizations in PNG format

---

## Design Choices

- **Storage**: JSON file for simplicity and transparency
- **Embedding Model**: `vertex_ai/gemini-embedding-001` for semantic similarity
- **Chat Model**: `vertex_ai/gemini-2.5-flash` for fact extraction
- **Retrieval**: Hybrid approach combining embedding similarity and keyword matching
- **Evaluation**: Automated pipeline with clear metrics and visualizations

---

## Limitations

- **Scalability**: JSON storage may not scale well for very large memory stores
- **Semantic Understanding**: The system relies on the underlying models for understanding context
- **Complex Reasoning**: Multi-hop reasoning is limited by the quality of extracted facts
- **Contradiction Detection**: Simple heuristic-based approach may miss subtle contradictions

---

## AI Usage Notes

This project was developed with the assistance of AI tools for:

- Generating synthetic conversation data
- Writing boilerplate code and tests
- Implementing standard metrics
- Creating visualizations
- Debugging and optimization

> The design decisions, system architecture, and interpretation of results were made by the human engineer.

# OpenGPU - infra_training Module

The `infra_training` directory is a key component of the **OpenGPU** project, which provides a modular pipeline for detecting available GPU hardware, automatically selecting compatible models, performing RAG (Retrieval-Augmented Generation), and supporting lightweight fine-tuning workflows.

This module serves as the **LLM training and orchestration layer** of the OpenGPU stack, handling everything from GPU environment validation to dataset ingestion, fine-tuning jobs, and inference serving.

---

## 🧠 Overview

The `infra_training` directory contains code responsible for:
1. **GPU Detection and Capability Analysis**
   - Identifies available GPUs, their memory, compute capability (SM version), and CUDA/ROCm compatibility.
   - Determines which model weights are feasible to load and train.
   - Exposes a simple interface for querying system capabilities before attempting to pull models.

2. **Model Acquisition and Initialization**
   - Automatically selects and downloads an appropriate model checkpoint (e.g., LLaMA, Mistral, or Falcon) depending on GPU memory.
   - Caches models locally for reuse.
   - Handles tokenizers, safetensors, and Hugging Face model integration.

3. **Retrieval-Augmented Generation (RAG) Integration**
   - Integrates document ingestion and embedding pipelines.
   - Builds a vector index for contextual augmentation of LLM responses.
   - Supports local retrieval backends like FAISS or ChromaDB.

4. **Fine-Tuning and Adapter Training**
   - Provides training utilities for LoRA, QLoRA, and full fine-tuning based on the selected model and system resources.
   - Reads from curated datasets (text, code, logs, etc.) gathered by the Rust cluster scraper binary.
   - Uses PEFT and Hugging Face Trainer or DeepSpeed configurations for scalable fine-tuning.

5. **Model Evaluation and Inference Serving**
   - Includes evaluation metrics (perplexity, BLEU, etc.) for tuned models.
   - Launches an inference endpoint for serving local responses, optionally integrated with RAG context fetching.

---

## ⚙️ Key Components

| File | Purpose |
|------|----------|
| `infra_learning.py` | Main entry point for training orchestration. Handles GPU detection, model loading, RAG setup, and fine-tuning control flow. |
| `model_loader.py` | Handles dynamic model selection based on GPU resources, caching, and tokenizer setup. |
| `rag_pipeline.py` | Defines the retrieval pipeline, vector store initialization, and context fetching logic. |
| `train_utils.py` | Training helpers for LoRA, dataset tokenization, checkpointing, and PEFT integration. |
| `data_ingest.py` | Reads and preprocesses datasets scraped by the Rust data harvester, cleans text, and formats input for fine-tuning. |
| `eval_utils.py` | Evaluation functions for tuned models, including loss, perplexity, and text quality metrics. |
| `config.yaml` | Central configuration for model IDs, cache paths, and runtime parameters. |

---

## 🔗 Integration with Rust Data Harvester

The **Rust binary** (`open_cluster_scraper`) runs across the GPU cluster to gather:
- System telemetry (GPU specs, uptime, utilization)
- Local datasets (logs, configs, structured/unstructured text)
- Network topology metadata

This data is streamed into the Python side of OpenGPU, where it is indexed and used for fine-tuning or RAG data enrichment.

---

## 🚀 Typical Workflow

1. **Cluster Detection**
   ```bash
   python infra_learning.py --detect
   ```
   → Detects GPUs, available VRAM, and capability score.

2. **Model Loading**
   ```bash
   python infra_learning.py --load
   ```
   → Loads a model checkpoint compatible with detected hardware.

3. **Fine-Tuning**
   ```bash
   python infra_learning.py --train --dataset ./datasets/cluster_texts.json
   ```

4. **RAG Augmented Inference**
   ```bash
   python infra_learning.py --serve --rag ./vector_index/
   ```

5. **Evaluation**
   ```bash
   python infra_learning.py --eval ./checkpoints/latest
   ```

---

## 🧩 Tech Stack

- **Languages:** Python, Rust  
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, FAISS  
- **Infra:** CUDA/ROCm, Kubernetes, Helm (for deployment)  
- **Data Storage:** Local cache + ChromaDB/FAISS vectors  
- **Purpose:** Efficient model orchestration, fine-tuning, and RAG integration for adaptive LLMs across heterogeneous GPU clusters

---

## 🧾 Notes

- Ensure proper ROCm/CUDA driver compatibility before running.
- Requires Python ≥ 3.10 and PyTorch with GPU support.
- Rust scraper should be built and running before fine-tuning jobs if RAG data ingestion is needed.

---

## 📁 Example Directory Layout

```
infra_training/
├── data_ingest.py
├── eval_utils.py
├── infra_learning.py
├── model_loader.py
├── rag_pipeline.py
├── train_utils.py
├── config.yaml
└── README.md  <-- this file
```

---

## 🧭 Future Directions

- Add automatic quantization and model distillation.
- Integrate distributed fine-tuning with DeepSpeed Zero-3.
- Expand Rust scraper telemetry (NVML stats, node health).
- Provide UI dashboard for RAG performance visualization.
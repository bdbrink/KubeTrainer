# 🧠 KubeTrainer
### *Distributed LLM training, fine-tuning, and RAG pipelines — natively on Kubernetes.*

---

## 🚀 Overview

**KubeTrainer** (formerly *OpenGPU*) is a cloud and vendor agnostic platform for orchestrating, training, and serving large language models (LLMs) directly within Kubernetes clusters.

It automatically detects available GPU resources, provisions compatible models, enables distributed fine-tuning with LoRA or full-precision modes, and integrates **retrieval-augmented generation (RAG)** to ground model outputs in real cluster data.

Built for platform engineers and AI practitioners who want to **bring model intelligence to their Kubernetes workloads** — no external GPU platforms required.

---

## ⚙️ Core Capabilities

| Component | Description |
|------------|--------------|
| 🧩 **GPU Detection & Scheduling** | Detects available GPU nodes, memory, compute capabilities, and runtime (CUDA/ROCm) to dynamically schedule model workloads. |
| 🧠 **Adaptive Model Loading** | Automatically selects and downloads the most capable model compatible with detected hardware (e.g., LLaMA, Mistral, Falcon). |
| 🔍 **Retrieval-Augmented Generation (RAG)** | Embeds structured/unstructured data from cluster logs, configs, and metrics for context-aware responses. |
| 🎓 **Fine-Tuning & Training** | Supports LoRA, QLoRA, and full fine-tuning modes using data gathered from the cluster. Uses PyTorch + Hugging Face PEFT. |
| ⚡ **Rust Data Harvester (Cluster Agent)** | Lightweight Rust binary that scrapes node telemetry, logs, and metadata to feed RAG and fine-tuning pipelines. |
| 🧰 **Inference & Evaluation** | Serves tuned models locally or across the cluster, with built-in evaluation metrics and caching layers. |

---

## 🏗️ Architecture Overview

```
             ┌──────────────────────────────────────┐
             │             KubeTrainer               │
             │--------------------------------------│
             │                                      │
             │  ┌───────────────┐                   │
             │  │ GPU Detector  │  → Detects GPUs   │
             │  └───────────────┘                   │
             │        ↓                             │
             │  ┌───────────────┐                   │
             │  │ Model Loader  │  → Loads LLMs     │
             │  └───────────────┘                   │
             │        ↓                             │
             │  ┌───────────────┐                   │
             │  │ RAG Pipeline  │  → Vector Search  │
             │  └───────────────┘                   │
             │        ↓                             │
             │  ┌───────────────┐                   │
             │  │ Trainer Engine│  → LoRA / PEFT    │
             │  └───────────────┘                   │
             │        ↓                             │
             │  ┌───────────────┐                   │
             │  │ Eval + Serve  │  → Inference API  │
             │  └───────────────┘                   │
             │                                      │
             └──────────────────────────────────────┘
                     ↑
         ┌───────────────────────────┐
         │ Rust Cluster Agent        │
         │ Collects data, logs, GPU  │
         │ stats → feeds into RAG    │
         └───────────────────────────┘
```

---

## 🧩 Repository Layout

```
kubetrainer/
├── infra_training/       # Core Python pipeline (training, RAG, orchestration)
│   ├── infra_learning.py
│   ├── rag_pipeline.py
│   ├── train_utils.py
│   └── model_loader.py
├── cluster_agent/        # Rust binary for telemetry + data harvesting
├── helm/                 # Helm chart for deploying KubeTrainer to K8s
├── scripts/              # Setup and helper scripts
└── README.md
```

---

## 🧠 Typical Workflow

1. **Cluster GPU Detection**
   ```bash
   python infra_training/infra_learning.py --detect
   ```

2. **Model Acquisition & Caching**
   ```bash
   python infra_training/infra_learning.py --load
   ```

3. **Fine-Tune the Model**
   ```bash
   python infra_training/infra_learning.py --train --dataset ./datasets/cluster_texts.json
   ```

4. **Enable RAG-Augmented Inference**
   ```bash
   python infra_training/infra_learning.py --serve --rag ./vector_index/
   ```

5. **Evaluate Results**
   ```bash
   python infra_training/infra_learning.py --eval ./checkpoints/latest
   ```

---

## 🧩 Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **Core** | Python, Rust |
| **LLM Framework** | PyTorch, Hugging Face Transformers, PEFT |
| **Retrieval** | FAISS / ChromaDB |
| **Infra** | Kubernetes, Helm |
| **GPU Runtimes** | CUDA / ROCm |
| **Orchestration** | K3s, Minikube, or full cluster deployment |

---

## 🧭 Roadmap

- [ ] Distributed fine-tuning with DeepSpeed Zero-3  
- [ ] Full cluster-wide RAG caching layer  
- [ ] Model quantization and distillation for edge GPUs  
- [ ] Node health dashboard for GPU usage telemetry  
- [ ] KubeTrainer Operator for CRD-based training jobs  

---

## 📜 License

Apache 2.0 — free for research and commercial use.

---

## ❤️ Acknowledgments

Built by **Brendan Brink**  
→ For engineers building intelligent, adaptive clusters.  

> “Train smarter. Scale faster. All within Kubernetes.”
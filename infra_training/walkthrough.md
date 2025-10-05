# SRE AI Training Pipeline

A complete machine learning pipeline for training Site Reliability Engineering (SRE) AI assistants with GPU detection, RAG (Retrieval-Augmented Generation), and fine-tuning capabilities.

## Overview

This pipeline consists of three integrated Python scripts that work together to:

1. **Detect hardware capabilities** (GPU/CPU) using Rust-based detection
2. **Load and process training data** from multiple formats (Markdown, JSON, JSONL)
3. **Build RAG systems** for context-aware responses
4. **Fine-tune language models** with LoRA for SRE-specific tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    sre_training.py                          │
│  • GPU detection (Rust integration)                         │
│  • Model loading with hardware optimization                 │
│  • Inference testing                                        │
│  • Launches RAG/fine-tuning pipeline                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ (saves model_info.pkl)
┌─────────────────────────────────────────────────────────────┐
│              sre_rag_finetuning.py                          │
│  • RAG system with FAISS vector database                    │
│  • LoRA-based fine-tuning                                   │
│  • Training data from training_data directory               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ (reads from)
┌─────────────────────────────────────────────────────────────┐
│              training_loader.py                             │
│  • Universal data loader (MD, JSON, JSONL)                  │
│  • Auto-format detection                                    │
│  • Sample file generation                                   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Dependencies

```bash
# Core ML libraries
pip install torch transformers

# RAG dependencies
pip install faiss-cpu sentence-transformers pandas

# Fine-tuning dependencies
pip install datasets peft trl

# System monitoring
pip install psutil
```

### Optional: Rust GPU Detection

For enhanced GPU detection (especially for AMD GPUs):

```bash
cd gpu-detect
cargo build --release
```

The script will fall back to PyTorch detection if the Rust binary isn't available.

## Quick Start

### 1. Create Training Data

Generate sample training files:

```bash
python3 training_loader.py
```

Or manually create files in `./training_data/`:

**Markdown (Q&A format):**
```markdown
## Q: How do I fix Kubernetes pods in CrashLoopBackOff?

**A:** Check logs with kubectl logs, verify resource limits, and review health probes.

## Q: What causes 503 errors?

**A:** Common causes include backend unavailability, timeout issues, or misconfigured health checks.
```

**JSON:**
```json
[
  {
    "instruction": "How to debug high memory usage?",
    "response": "Start with top/htop, check for memory leaks, review heap dumps..."
  }
]
```

**JSONL:**
```jsonl
{"instruction": "Explain circuit breakers", "response": "Circuit breakers prevent cascading failures..."}
{"instruction": "How to monitor Kubernetes?", "response": "Use Prometheus, Grafana, and distributed tracing..."}
```

### 2. Run the Training Pipeline

**Full pipeline (GPU detection → Model loading → RAG → Fine-tuning):**

```bash
python3 sre_training.py
```

The script will:
- Detect your GPU/CPU capabilities
- Load an appropriately-sized model
- Run inference tests
- Save model info
- Prompt you to launch RAG/fine-tuning

**Manual RAG/Fine-tuning launch:**

```bash
# RAG only
python3 sre_rag_finetuning.py --mode rag --test

# Fine-tuning only
python3 sre_rag_finetuning.py --mode finetune --model-info ./model_info.pkl

# Both
python3 sre_rag_finetuning.py --mode both --model-info ./model_info.pkl --test
```

### 3. Create Sample Training Files

```bash
python3 sre_rag_finetuning.py --create-samples
```

## Detailed Usage

### Training Data Formats

The `training_loader.py` supports multiple formats with auto-detection:

**Markdown Formats:**
- Section format (headers + content)
- Q&A format (`Q:` and `A:` markers)
- Conversational format (`User:` and `Assistant:`)

**JSON Formats:**
- Array of objects
- Nested structure with `examples` key
- Various field names (instruction/response, question/answer, prompt/completion)

**JSONL Format:**
- One JSON object per line
- Supports same field variations as JSON

### Hardware Optimization

The pipeline automatically optimizes for your hardware:

**GPU Detection:**
- Uses Rust binary for accurate AMD/NVIDIA detection
- Falls back to PyTorch CUDA detection
- CPU-only fallback

**Model Selection:**
- VRAM ≥ 20GB: Qwen2-14B-Instruct
- VRAM ≥ 12GB: Qwen2-7B-Instruct
- VRAM ≥ 8GB: Qwen2-1.5B-Instruct
- Default: Qwen2-0.5B-Instruct

**Batch Size:**
- Dynamically calculated based on VRAM
- Conservative settings for AMD GPUs
- Gradient accumulation for memory efficiency

**AMD GPU Support:**
- Conservative batch sizes
- Disabled mixed precision (float32 only)
- Progressive fallback strategies
- HIP error handling

### RAG System

**Features:**
- FAISS vector database for fast similarity search
- Sentence-BERT embeddings (all-MiniLM-L6-v2)
- Automatic chunking with overlap
- Context-aware prompt generation

**Configuration:**
```python
RAGConfig(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    chunk_overlap=50,
    top_k=5
)
```

**Search Example:**
```python
rag = RAGSystem(config, device="cuda")
results = rag.search("How to fix 503 errors?", top_k=5)
prompt = rag.generate_rag_prompt("How to fix 503 errors?")
```

### Fine-Tuning

**LoRA Configuration:**
- Rank: 8 (memory-efficient)
- Target modules: q_proj, v_proj (attention only)
- Gradient checkpointing enabled
- Conservative batch sizes

**Memory Optimization:**
- Halved batch size from base recommendation
- 8-step gradient accumulation
- Single checkpoint retention
- Aggressive memory cleanup

**Training Arguments:**
```python
TrainingArguments(
    per_device_train_batch_size=effective_batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    gradient_checkpointing=True
)
```

## Command-Line Options

### sre_training.py

```bash
python3 sre_training.py
# Interactive - detects GPU, loads model, prompts for next steps
```

### sre_rag_finetuning.py

```bash
# RAG only
python3 sre_rag_finetuning.py --mode rag --test

# Fine-tuning only  
python3 sre_rag_finetuning.py --mode finetune --model-info ./model_info.pkl

# Both with custom directories
python3 sre_rag_finetuning.py \
  --mode both \
  --model-info ./model_info.pkl \
  --training-data ./my_training_data \
  --output-dir ./my_outputs \
  --test

# Create sample files
python3 sre_rag_finetuning.py --create-samples
```

### training_loader.py

```bash
# Generate samples and test loading
python3 training_loader.py
```

## Output Structure

```
.
├── training_data/          # Your training files (.md, .json, .jsonl)
├── rag_vectors/            # FAISS vector database
│   ├── vectors.faiss
│   └── metadata.pkl
├── sre_outputs/            # Training outputs
│   └── fine_tuned_model/   # LoRA adapters
├── model_info.pkl          # Serialized model info
└── knowledge_base/         # RAG metadata
```

## Troubleshooting

### AMD GPU Issues

**HIP Errors:**
- Pipeline uses float32 (not mixed precision)
- Conservative batch sizes
- Progressive fallback strategies
- Disable attention optimizations

**If GPU fails:**
- Script automatically falls back to CPU
- Smaller model auto-selected
- Training still completes

### Memory Issues

**Reduce memory usage:**
1. Use smaller model (edit model selection logic)
2. Decrease batch size in code
3. Increase gradient accumulation steps
4. Disable gradient checkpointing (not recommended)

### No Training Data

**Error: No training data found**

```bash
# Generate samples
python3 sre_rag_finetuning.py --create-samples

# Or add your own files to ./training_data/
```

## Advanced Usage

### Custom Model Selection

Edit `sre_training.py`:

```python
def get_recommended_model_size(self):
    return ("custom", "your-org/your-model-name")
```

### Custom RAG Configuration

Edit `sre_rag_finetuning.py`:

```python
rag_config = RAGConfig(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    chunk_size=1024,
    top_k=10
)
```

### Training Data Validation

```python
from training_loader import UniversalTrainingLoader

loader = UniversalTrainingLoader("./training_data")
examples = loader.load_all_data()

# Inspect loaded data
for ex in examples:
    print(f"Q: {ex['instruction']}")
    print(f"A: {ex['response']}\n")
```

## Performance Tips

1. **Use JSONL for large datasets** - more memory efficient than JSON
2. **Keep markdown files focused** - one topic per file
3. **Use meaningful filenames** - helps with debugging
4. **Test with --test flag first** - validates setup before heavy training
5. **Monitor GPU memory** - watch for OOM errors during fine-tuning

## Known Limitations

- AMD GPU support is experimental (ROCm compatibility varies)
- FAISS indexing requires sufficient RAM for large datasets
- Fine-tuning requires model to fit in available memory
- No distributed training support (single GPU/CPU only)

## License & Credits

This pipeline integrates:
- Transformers (Hugging Face)
- PEFT/LoRA (Hugging Face)
- FAISS (Meta AI)
- Sentence-BERT (UKP Lab)
- Qwen models (Alibaba Cloud)
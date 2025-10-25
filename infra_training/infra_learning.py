#!/usr/bin/env python3
"""
SRE AI Training Script - Enhanced with Rust GPU Detection Integration
FIXED: AMD GPU compatibility and method signature issues
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
import subprocess
import json
import sys
import pickle
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class ModelCandidate:
    """Represents a model candidate from HuggingFace"""
    model_id: str
    size_gb: float
    downloads: int
    tags: List[str]
    
    def __repr__(self):
        return f"{self.model_id} ({self.size_gb:.1f}GB, {self.downloads:,} downloads)"

class HuggingFaceModelSelector:
    """Dynamic model selection from HuggingFace API"""
    
    def __init__(self):
        self.api_base = "https://huggingface.co/api/models"
        self.cached_model = None
        self.cached_reason = None
        self._cached_candidates = None
    
    def search_sre_models(self, max_results: int = 30) -> List[ModelCandidate]:
        """Search HuggingFace for reasoning/systems models suitable for SRE tasks"""

        # Return cached results if available
        if self._cached_candidates is not None:
            print("📦 Using cached model candidates")
            return self._cached_candidates

        print("🔍 Querying HuggingFace for reasoning & systems models...")
        
        # Blacklist patterns for problematic models
        BLACKLIST_PATTERNS = [
            'gguf',           # GGUF quantized (use llama.cpp instead)
            'abliterated',    # Custom merged models
            'uncensored',     # Often custom/unstable
            'franken',        # Frankenstein merges
            'moe',            # Many custom MOEs lack proper configs
            'gated-moe',      # Custom gated MOEs
            'exl2',           # ExLlamaV2 quantization
            'awq',            # AWQ quantization (unless you have AutoAWQ)
            'gptq',           # GPTQ quantization (unless you have AutoGPTQ)
            'fp8',            # FP8 quantization (H100+ only)
            'int8',           # INT8 quantization (needs bitsandbytes)
            'int4',           # INT4 quantization (needs bitsandbytes)
            'bnb',            # ✨ bitsandbytes quantization
            '4bit',           # ✨ 4-bit quantization
            '8bit',           # ✨ 8-bit quantization
            'nf4',            # ✨ NormalFloat 4-bit
            'unsloth',        # ✨ Unsloth optimized models (usually quantized)
        ]
        
        search_queries = [
            "deepseek-r1 instruct",
            "qwen2.5 reasoning", 
            "llama-3 instruct",
            "mistral instruct"
        ]
        
        all_candidates = []
        seen_ids = set()
        
        for search_term in search_queries:
            params = {
                "search": search_term,
                "filter": "text-generation",
                "sort": "downloads",
                "direction": -1,
                "limit": max_results
            }
            
            try:
                response = requests.get(self.api_base, params=params, timeout=15)
                response.raise_for_status()
                models = response.json()
                
                for model in models:
                    model_id = model.get('modelId', '')
                    tags = model.get('tags', [])
                    
                    # Skip duplicates
                    if model_id in seen_ids:
                        continue
                    
                    # Skip blacklisted patterns
                    model_lower = model_id.lower()
                    if any(pattern in model_lower for pattern in BLACKLIST_PATTERNS):
                        print(f"⚠️  Skipping incompatible model: {model_id}")
                        continue
                    
                    # Prefer official/org models over user uploads
                    is_official = '/' in model_id and not model_id.split('/')[0].startswith(('David', 'User', 'TheBloke'))
                    
                    # Filter for reasoning/analysis-capable models
                    is_suitable = any(kw in model_lower or kw in str(tags).lower() 
                                    for kw in [
                                        'r1', 'reasoning', 'deepseek',
                                        'qwen2.5', 'llama-3', 'mistral',
                                        'instruct', 'chat'
                                    ])
                    
                    if not is_suitable:
                        continue
                    
                    seen_ids.add(model_id)
                    size_gb = self._estimate_size(model_id)
                    
                    # Boost official models
                    downloads = model.get('downloads', 0)
                    if is_official:
                        downloads *= 1.5
                    
                    all_candidates.append(ModelCandidate(
                        model_id=model_id,
                        size_gb=size_gb,
                        downloads=int(downloads),
                        tags=tags
                    ))
                
            except Exception as e:
                print(f"⚠️  Query '{search_term}' failed: {e}")
                continue
        
        # Sort candidates
        all_candidates = sorted(all_candidates, key=lambda x: x.downloads, reverse=True)
        
        # Cache the results (fallback to curated if empty)
        if all_candidates:
            self._cached_candidates = all_candidates
        else:
            self._cached_candidates = self._get_curated_models()
        
        # Now it's safe to print
        print(f"✅ Found {len(self._cached_candidates)} compatible SRE models")
        
        return self._cached_candidates
    
    def _estimate_size(self, model_id: str) -> float:
        """Estimate model size from ID"""
        model_id_lower = model_id.lower()
        
        # Size patterns in GB (accounting for fp16 - 2 bytes per param)
        # Sorted by size descending to match larger models first
        patterns = {
            '405b': 810, '400b': 800,
            '175b': 350, '176b': 352,
            '70b': 140, '65b': 130, '60b': 120,
            '34b': 68, '33b': 66, '32b': 64,
            '30b': 60, '20b': 40,
            '15b': 30, '14b': 28, '13b': 26,
            '12b': 24, '11b': 22, '10b': 20,
            '8b': 16, '7b': 14, '6.7b': 13.4,
            '3b': 6, '2.7b': 5.4, '2b': 4,
            '1.5b': 3, '1.3b': 2.6, '1b': 2,
            '0.5b': 1, '500m': 1
        }

        
        for pattern, size in patterns.items():
            if pattern in model_id_lower:
                return size
        
        return 6.0  # Default assumption
    
    def _get_curated_models(self) -> List[ModelCandidate]:
        """Curated fallback list of proven coding models"""
        return [
            # Top tier coding models
            ModelCandidate("deepseek-ai/deepseek-coder-33b-instruct", 66, 400000, ["code"]),
            ModelCandidate("WizardLM/WizardCoder-33B-V1.1", 66, 300000, ["code"]),
            ModelCandidate("codellama/CodeLlama-34b-Instruct-hf", 68, 500000, ["code"]),
            
            # Mid-tier (12-16GB VRAM)
            ModelCandidate("deepseek-ai/deepseek-coder-6.7b-instruct", 13.4, 600000, ["code"]),
            ModelCandidate("WizardLM/WizardCoder-15B-V1.0", 30, 400000, ["code"]),
            ModelCandidate("codellama/CodeLlama-13b-Instruct-hf", 26, 450000, ["code"]),
            
            # General purpose with strong coding
            ModelCandidate("Qwen/Qwen2.5-Coder-7B-Instruct", 14, 500000, ["code"]),
            ModelCandidate("Qwen/Qwen2-7B-Instruct", 14, 700000, ["code"]),
            
            # Smaller models (4-8GB VRAM)
            ModelCandidate("deepseek-ai/deepseek-coder-1.3b-instruct", 2.6, 300000, ["code"]),
            ModelCandidate("Qwen/Qwen2-1.5B-Instruct", 3, 400000, ["general"]),
        ]
    
    
    def _get_actual_model_size(self, model_id: str) -> float:
        """Get actual model size from HuggingFace API"""
        try:
            # Get model info from HuggingFace API
            response = requests.get(
                f"https://huggingface.co/api/models/{model_id}",
                timeout=10
            )
            if response.ok:
                data = response.json()
                
                # Method 1: Check if safetensors field exists (indicates safetensors files)
                if data.get('safetensors'):
                    try:
                        # Get the safetensors file info
                        response = requests.get(
                            f"https://huggingface.co/api/models/{model_id}/revision/main",
                            timeout=10
                        )
                        if response.ok:
                            files = response.json()
                            total_size = sum(
                                file.get('size', 0)
                                for file in files.get('siblings', [])
                                if file.get('rfilename', '').endswith(('.safetensors', '.bin'))
                            )
                            if total_size > 0:
                                return total_size / (1024**3)  # Convert to GB
                    except Exception:
                        pass
                
                # Method 2: Try to get size from model config if available
                if data.get('siblings'):
                    total_size = sum(
                        file.get('size', 0)
                        for file in data.get('siblings', [])
                        if file.get('rfilename', '').endswith(('.safetensors', '.bin', '.pt'))
                    )
                    if total_size > 0:
                        return total_size / (1024**3)  # Convert to GB
        
        except Exception as e:
            print(f"Error fetching model size for {model_id}: {e}")
        
        # Fall back to estimation if API methods fail
        return self._estimate_size(model_id)

    def recommend_for_hardware(self, vram_gb: float, gpu_type: str) -> Tuple[str, str]:
        """Get best model for hardware specs"""
        # DISABLED: Don't use cache, always recalculate with latest constraints
        # if self.cached_model is not None:
        #     print("\n📦 Using cached model recommendation")
        #     return self.cached_model, self.cached_reason

        candidates = self.search_sre_models()
        
        # AMD needs more safety margin due to ROCm overhead
        safety = 0.55 if 'amd' in gpu_type.lower() else 0.75
        usable_vram = vram_gb * safety

        # Get actual sizes for candidates with better error handling
        print("🔍 Checking actual model sizes...")
        for candidate in candidates[:10]:
            actual_size = self._get_actual_model_size(candidate.model_id)
            if actual_size > 0 and actual_size != candidate.size_gb:
                print(f"📏 {candidate.model_id}: {actual_size:.1f}GB (was {candidate.size_gb:.1f}GB)")
                candidate.size_gb = actual_size
            elif actual_size <= 0:
                # Estimation failed, re-estimate more carefully
                candidate.size_gb = self._estimate_size(candidate.model_id)
                print(f"⚠️ {candidate.model_id}: Using estimated {candidate.size_gb:.1f}GB")
        
        # CRITICAL: Account for fp32 (2x size) and loading overhead (1.5x)
        # For AMD GPU, we use fp32, so model size * 2 * 1.5 = 3x the base size
        is_amd = 'amd' in gpu_type.lower()
        size_multiplier = 2.0 if is_amd else 1.5  # fp32 + overhead for AMD, just overhead for NVIDIA
        
        # Filter models that fit with loading overhead
        fitting = [m for m in candidates if (m.size_gb * size_multiplier) <= usable_vram]
        
        if not fitting:
            # If nothing fits, find the absolute smallest
            smallest = min(candidates, key=lambda m: m.size_gb)
            print(f"\n⚠️ No models fit in {usable_vram:.1f}GB usable VRAM")
            print(f"   Returning smallest available: {smallest.model_id} ({smallest.size_gb:.1f}GB)")
            return (smallest.model_id, f"Smallest available ({smallest.size_gb:.1f}GB base)")
        
        # Sort by size (bigger is better, if it fits)
        fitting.sort(key=lambda m: (m.size_gb, m.downloads), reverse=True)
        
        best = fitting[0]
        
        # Show options
        print(f"\n📊 Top models for {vram_gb:.1f}GB VRAM ({gpu_type}):")
        print(f"   Usable VRAM after safety margin: {usable_vram:.1f}GB")
        print(f"   Size multiplier for loading: {size_multiplier}x")
        print()
        for i, model in enumerate(fitting[:5], 1):
            marker = "👉" if i == 1 else "  "
            needed_vram = model.size_gb * size_multiplier
            print(f"{marker} {i}. {model.model_id}")
            print(f"      Base: {model.size_gb:.1f}GB | Needs: {needed_vram:.1f}GB VRAM | Downloads: {model.downloads:,}")
        
        print(f"\n✅ Auto-selected: {best.model_id}")
        try:
            choice = input("Press Enter to accept, or enter number (1-5) to choose different model: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(fitting):
                    best = fitting[idx]
                    print(f"✅ Using: {best.model_id}")
        except (KeyboardInterrupt, EOFError):
            pass
        
        needed_vram = best.size_gb * size_multiplier
        reason = f"{best.size_gb:.1f}GB base (needs {needed_vram:.1f}GB with overhead), {best.downloads:,} downloads"
        
        # Cache the result
        self.cached_model = best.model_id
        self.cached_reason = reason

        return self.cached_model, self.cached_reason

class GPUManager:
    """Interface between Rust GPU detection and Python training logic"""
    
    def __init__(self, rust_binary_path: Optional[str] = None):
        # Try to auto-detect the Rust binary location based on your project structure
        if rust_binary_path is None:
            possible_paths = [
                "../gpu-detect/target/release/gpu-detect"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    rust_binary_path = path
                    print(f"Found Rust binary at: {path}")
                    break
            
            if rust_binary_path is None:
                rust_binary_path = "../gpu-detect/target/release/gpu_detect"
        
        self.rust_binary_path = rust_binary_path
        self.gpu_info = None
        self.system_specs = None
        self._cached_model_recommendation = None
        self.model_selector = HuggingFaceModelSelector()
        self._detect_gpu_info()
    
    def _detect_gpu_info(self) -> None:
        """Run Rust GPU detection and parse results"""
        print("🔍 Running Rust GPU Detection...")
        
        try:
            # Check if Rust binary exists
            if not Path(self.rust_binary_path).exists():
                print(f"❌ Rust binary not found at {self.rust_binary_path}")
                print("💡 Run 'cargo build --release' to build the GPU detection tool")
                self._fallback_to_pytorch_detection()
                return
            
            # Run the Rust GPU detection - redirect stderr to suppress debug output
            result = subprocess.run(
                [self.rust_binary_path, "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            
            if result.stdout.strip():
                try:
                    # The output contains debug logs, but JSON should be the last complete line
                    output_lines = result.stdout.strip().split('\n')
                    
                    # Find the line that looks like JSON (starts with { and ends with })
                    json_line = None
                    for line in reversed(output_lines):
                        stripped = line.strip()
                        if stripped.startswith('{') and stripped.endswith('}'):
                            json_line = stripped
                            break
                    
                    if json_line:
                        self.gpu_info = json.loads(json_line)
                        print(f"✅ GPU Detection successful via Rust")
                        self._print_gpu_summary()
                    else:
                        print("⚠️ No valid JSON line found, using fallback")
                        self._fallback_to_pytorch_detection()
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    print(f"Raw output: {repr(result.stdout)}")
                    self._fallback_to_pytorch_detection()
            else:
                print("⚠️ No output from Rust detection, using fallback")
                self._fallback_to_pytorch_detection()
                
        except subprocess.TimeoutExpired:
            print("⚠️ Rust GPU detection timed out, using fallback")
            self._fallback_to_pytorch_detection()
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Rust GPU detection failed: {e}")
            self._fallback_to_pytorch_detection()
        except FileNotFoundError:
            print(f"⚠️ Rust binary not found at {self.rust_binary_path}")
            self._fallback_to_pytorch_detection()
        
    def _fallback_to_pytorch_detection(self) -> None:
        """Fallback to PyTorch's built-in GPU detection"""
        print("🔄 Using PyTorch GPU detection as fallback...")
        
        self.gpu_info = {
            "gpu_type": "CPU Only",
            "vram_gb": 0.0,
            "is_ml_ready": False,
            "compute_capability": None,
            "driver_version": None
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.gpu_info = {
                "gpu_type": f"NVIDIA {torch.cuda.get_device_name(0)}",
                "vram_gb": props.total_memory / (1024**3),
                "is_ml_ready": props.total_memory > 2147483648,  # 2GB+
                "compute_capability": f"{props.major}.{props.minor}",
                "driver_version": None
            }
    
    def _print_gpu_summary(self) -> None:
        """Print a summary of detected GPU info"""
        if self.gpu_info:
            print(f"📊 GPU: {self.gpu_info.get('gpu_type', 'Unknown')}")
            if self.gpu_info.get('vram_gb', 0) > 0:
                print(f"📊 VRAM: {self.gpu_info['vram_gb']:.1f}GB")
            print(f"📊 ML Ready: {'Yes' if self.gpu_info.get('is_ml_ready', False) else 'No'}")
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on GPU/system capabilities"""
        if not self.gpu_info or not self.gpu_info.get('is_ml_ready', False):
            return 16  # Conservative for CPU or limited GPU
        
        vram_gb = self.gpu_info.get('vram_gb', 0)
        gpu_type = self.gpu_info.get('gpu_type', '').lower()
        
        # More conservative batch sizes for AMD GPU to avoid HIP errors
        if 'amd' in gpu_type or 'radeon' in gpu_type:
            # AMD GPUs need smaller batch sizes due to ROCm limitations
            if vram_gb >= 16:
                return 32  # Conservative for your 7800 XT
            elif vram_gb >= 12:
                return 24
            elif vram_gb >= 8:
                return 16
            else:
                return 8
        
        # Original logic for NVIDIA
        if vram_gb >= 24:
            return 128  # Large batch for high-end cards
        elif vram_gb >= 16:
            return 64
        elif vram_gb >= 12:
            return 48
        elif vram_gb >= 8:
            return 32
        elif vram_gb >= 6:
            return 24
        else:
            return 16
    
    def should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision training should be used"""
        if not self.gpu_info or not self.gpu_info.get('is_ml_ready', False):
            return False
        
        gpu_type = self.gpu_info.get('gpu_type', '').lower()
        
        # Be more conservative with AMD GPUs due to ROCm issues
        if 'amd' in gpu_type or 'radeon' in gpu_type:
            return False  # Disable mixed precision for AMD to avoid HIP errors
        
        # Enable for modern NVIDIA GPUs
        modern_nvidia = any(x in gpu_type for x in ['rtx', 'gtx 16', 'tesla', 'quadro rtx'])
        return modern_nvidia
    
    def get_recommended_model_size(self) -> Tuple[str, str]:
        """Get recommended model with dynamic HuggingFace lookup"""
        if not self.gpu_info:
            return ("small", "deepseek-ai/deepseek-coder-1.3b-instruct")
        
        vram_gb = self.gpu_info.get('vram_gb', 0)
        is_ml_ready = self.gpu_info.get('is_ml_ready', False)
        gpu_type = self.gpu_info.get('gpu_type', 'cpu')
        
        if not is_ml_ready or vram_gb < 3:
            return ("small", "deepseek-ai/deepseek-coder-1.3b-instruct")
        
        # Use dynamic selector
        print(f"\n🤖 Dynamic Model Selection")
        print("=" * 50)
        
        # Use the cached selector instead of creating a new one
        model_id, reason = self.model_selector.recommend_for_hardware(vram_gb, gpu_type)
        
        print(f"\n💡 Reason: {reason}")
        
        # Determine size category for logging
        if vram_gb >= 20:
            size_cat = "xlarge"
        elif vram_gb >= 12:
            size_cat = "large"
        elif vram_gb >= 8:
            size_cat = "medium"
        else:
            size_cat = "small"
        
        self._cached_model_recommendation = (size_cat, model_id)
        return self._cached_model_recommendation
    
    def get_torch_device_config(self) -> Dict:
        """Get PyTorch device configuration based on detected hardware"""
        config = {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True
        }
        
        if self.gpu_info and self.gpu_info.get('is_ml_ready', False):
            gpu_type = self.gpu_info.get('gpu_type', '').lower()
            
            if 'nvidia' in gpu_type and torch.cuda.is_available():
                config.update({
                    "device": "cuda",
                    "torch_dtype": torch.bfloat16 if self.should_use_mixed_precision() else torch.float16,
                    "device_map": "auto"
                })
            elif ('amd' in gpu_type or 'radeon' in gpu_type):
                # AMD GPU detected - try to use it but with very conservative settings
                if torch.cuda.is_available():  # ROCm uses CUDA API
                    print("🔧 AMD GPU detected - attempting to use with ultra-conservative settings")
                    config.update({
                        "device": "cuda",
                        "torch_dtype": torch.float32,  # Always float32 for AMD
                        "device_map": None,  # Never use auto device mapping
                        "low_cpu_mem_usage": True,
                        "attn_implementation": None  # Disable attention optimizations
                    })
                else:
                    print("⚠️ AMD GPU detected but ROCm not available, using CPU")
        
        return config
    
    def is_amd_gpu(self) -> bool:
        """Check if detected GPU is AMD"""
        if not self.gpu_info:
            return False
        gpu_type = self.gpu_info.get('gpu_type', '').lower()
        return 'amd' in gpu_type or 'radeon' in gpu_type

def enhanced_system_check(gpu_manager: GPUManager):
    """Enhanced system check using Rust GPU detection results"""
    print("🔍 Enhanced System Check (Rust + Python)")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # GPU info from Rust detection
    if gpu_manager.gpu_info:
        print(f"GPU (Rust): {gpu_manager.gpu_info['gpu_type']}")
        if gpu_manager.gpu_info.get('vram_gb', 0) > 0:
            print(f"VRAM (Rust): {gpu_manager.gpu_info['vram_gb']:.1f}GB")
        if gpu_manager.gpu_info.get('compute_capability'):
            print(f"Compute Capability: {gpu_manager.gpu_info['compute_capability']}")
        print(f"ML Ready: {gpu_manager.gpu_info.get('is_ml_ready', False)}")
        
        # AMD-specific warnings
        if gpu_manager.is_amd_gpu():
            print("⚠️ AMD GPU detected - using conservative settings to avoid HIP errors")
    
    # System specs
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    # Training recommendations
    print(f"\n📊 Training Recommendations:")
    print(f"Optimal Batch Size: {gpu_manager.get_optimal_batch_size()}")
    print(f"Mixed Precision: {gpu_manager.should_use_mixed_precision()}")
    
    model_size, model_name = gpu_manager.get_recommended_model_size()
    print(f"Recommended Model: {model_name} ({model_size})")
    print()

def load_model_with_gpu_config(gpu_manager: GPUManager):
    """Load model using GPU manager's recommended configuration"""
    print("\n🤖 Loading Model with Optimized Configuration")
    print("=" * 60)
    
    model_size, model_name = gpu_manager.get_recommended_model_size()
    device_config = gpu_manager.get_torch_device_config()

    system_ram_gb = psutil.virtual_memory().available / (1024**3)
    selector = HuggingFaceModelSelector()
    estimated_size = selector._estimate_size(model_name)
    
    # For fp32, we need 2x the fp16 size
    if device_config['torch_dtype'] == torch.float32:
        estimated_size *= 2
    
    print(f"Selected Model: {model_name}")
    print(f"Estimated Size: {estimated_size:.1f}GB ({device_config['torch_dtype']})")
    print(f"Available RAM: {system_ram_gb:.1f}GB")
    print(f"Device Config: {device_config['device']}")
    
    # Check if we have enough RAM
    if estimated_size * 1.3 > system_ram_gb:  # Need 30% overhead
        print(f"\n❌ CRITICAL: Not enough RAM!")
        print(f"   Model needs: ~{estimated_size * 1.3:.1f}GB")
        print(f"   Available: {system_ram_gb:.1f}GB")
        print(f"   Shortfall: {(estimated_size * 1.3 - system_ram_gb):.1f}GB")
        
        # Try to find a smaller model
        print(f"\n🔍 Searching for smaller model...")
        vram_gb = gpu_manager.gpu_info.get('vram_gb', 0)
        is_amd = gpu_manager.is_amd_gpu()
        safety = 0.55 if is_amd else 0.75
        
        # Look for models that fit in RAM
        candidates = selector.search_sre_models()
        
        for candidate in sorted(candidates, key=lambda x: x.size_gb):
            ram_needed = candidate.size_gb * 2 * 1.3  # fp32 + overhead
            if ram_needed < system_ram_gb and candidate.size_gb < vram_gb * safety:
                print(f"✅ Found alternative: {candidate.model_id} ({candidate.size_gb:.1f}GB)")
                model_name = candidate.model_id
                estimated_size = candidate.size_gb * 2  # fp32
                break
        else:
            print(f"❌ No suitable model found for available RAM")
            return None, None, None, None
    
    if gpu_manager.is_amd_gpu():
        print("🔧 AMD GPU: Attempting GPU loading with conservative settings")
        print("💡 If loading fails, we'll try progressively safer approaches")
    
    start_time = time.time()
    
    try:
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"🧠 Loading model on {device_config['device']}...")
        
        if gpu_manager.is_amd_gpu() and device_config['device'] == 'cuda':
            print("🔧 AMD GPU: Loading directly to GPU (no device_map)...")
            try:
                # For AMD, load directly to GPU without device_map
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,  # Don't use auto device mapping
                    trust_remote_code=True,
                    low_cpu_mem_usage=False  # Load directly
                )
                
                # Now move to GPU
                print("🔧 Moving model to AMD GPU...")
                model = model.to('cuda')
                actual_device = 'cuda'
                print("✅ Successfully loaded to AMD GPU!")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ OOM on AMD GPU - model too large for VRAM")
                    print("🔧 Falling back to CPU...")
                    # Reload on CPU
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    actual_device = 'cpu'
                else:
                    raise e
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=device_config['torch_dtype'],
                device_map=device_config['device_map'],
                trust_remote_code=True,
                low_cpu_mem_usage=device_config['low_cpu_mem_usage']
            )
            actual_device = device_config['device']
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return tokenizer, model, actual_device, model_name
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        
        # Try smallest possible fallback
        print("💡 Trying smallest available model...")
        fallback_model = "Qwen/Qwen2-0.5B-Instruct"
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Try GPU if AMD
            if gpu_manager.is_amd_gpu() and torch.cuda.is_available():
                try:
                    model = model.to('cuda')
                    print(f"✅ Fallback model loaded on AMD GPU!")
                    return tokenizer, model, "cuda", fallback_model
                except:
                    pass
            
            print(f"✅ Fallback model loaded on CPU")
            return tokenizer, model, "cpu", fallback_model
                
        except Exception as fallback_error:
            print(f"❌ All fallbacks failed: {fallback_error}")
            return None, None, None, None

def optimized_inference_test(tokenizer, model, device, gpu_manager: GPUManager):
    """Run inference test with progressive AMD GPU fallback strategies"""
    print("\n🚀 Optimized Inference Test")
    print("=" * 50)
    
    # SRE-themed test prompt
    test_prompt = """You are an SRE AI assistant. A web service shows 503 errors and high memory usage. What steps would you take to diagnose this?"""
    
    print(f"Prompt: {test_prompt}")
    print(f"Using batch size: {gpu_manager.get_optimal_batch_size()}")
    print(f"Mixed precision: {gpu_manager.should_use_mixed_precision()}")
    
    if gpu_manager.is_amd_gpu() and device == "cuda":
        print("🔥 AMD GPU: Attempting inference on GPU!")
    
    print(f"Running on: {device}")
    print("\nResponse:")
    print("-" * 30)
    
    # Progressive fallback strategies for AMD GPU
    strategies = [
        ("conservative", {"max_new_tokens": 100, "do_sample": False, "temperature": None, "top_p": None}),
        ("ultra_conservative", {"max_new_tokens": 50, "do_sample": False, "temperature": None, "top_p": None}),
        ("minimal", {"max_new_tokens": 25, "do_sample": False, "temperature": None, "top_p": None})
    ]
    
    # If not AMD or if on CPU, use normal strategy
    if not gpu_manager.is_amd_gpu() or device == "cpu":
        strategies = [("normal", {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        })]
    
    for strategy_name, generation_params in strategies:
        try:
            print(f"🔧 Trying {strategy_name} generation strategy...")
            
            # Tokenize inputs
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            # Move inputs to device
            if device == "cuda":
                try:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception as e:
                    print(f"⚠️ Failed to move inputs to GPU: {e}")
                    if gpu_manager.is_amd_gpu():
                        print("🔧 Moving model to CPU for inference...")
                        model = model.to('cpu')
                        device = 'cpu'
                    else:
                        raise e
            
            # Base generation config
            generation_config = {
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "return_dict_in_generate": False
            }
            
            # Add strategy-specific params
            generation_config.update(generation_params)
            
            # AMD-specific optimizations
            if gpu_manager.is_amd_gpu() and device == "cuda":
                # Force single-threaded generation for AMD
                torch.set_num_threads(1)
                # Disable some optimizations
                generation_config.update({
                    "num_beams": 1,
                    "early_stopping": False,
                    "output_scores": False,
                    "output_attentions": False,
                    "output_hidden_states": False
                })
            
            inference_start = time.time()
            
            # Try inference with current strategy
            with torch.no_grad():
                try:
                    # Set environment variable to help with HIP errors
                    if gpu_manager.is_amd_gpu():
                        os.environ['AMD_SERIALIZE_KERNEL'] = '3'
                        os.environ['HIP_VISIBLE_DEVICES'] = '0'
                    
                    outputs = model.generate(**inputs, **generation_config)
                    
                except RuntimeError as runtime_error:
                    error_msg = str(runtime_error).lower()
                    if "hip" in error_msg or "invalid device function" in error_msg:
                        print(f"❌ HIP error with {strategy_name} strategy: {runtime_error}")
                        if strategy_name == "minimal":
                            # Last resort: move to CPU
                            print("🔧 Final fallback: Moving to CPU...")
                            model = model.to('cpu')
                            inputs = {k: v.to('cpu') for k, v in inputs.items()}
                            device = 'cpu'
                            outputs = model.generate(**inputs, **generation_config)
                        else:
                            # Try next strategy
                            continue
                    else:
                        # Non-HIP error, re-raise
                        raise runtime_error
            
            # Success! Process the output
            inference_time = time.time() - inference_start
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(test_prompt):].strip()
            
            print(response)
            print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
            print(f"📊 Tokens/second: ~{generation_config['max_new_tokens']/inference_time:.1f}")
            
            if gpu_manager.is_amd_gpu():
                if device == "cuda":
                    print("🔥 SUCCESS: AMD GPU inference worked!")
                    print(f"💡 Successful strategy: {strategy_name}")
                else:
                    print("💡 AMD GPU: Inference completed on CPU after GPU issues")
            
            return True
            
        except Exception as e:
            print(f"❌ Strategy {strategy_name} failed: {e}")
            if strategy_name == strategies[-1][0]:  # Last strategy
                print("❌ All strategies failed")
                if gpu_manager.is_amd_gpu():
                    print("💡 AMD GPU compatibility issues detected")
                    print("💡 Suggestions:")
                    print("   - Check ROCm installation: rocm-smi")
                    print("   - Update PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6")
                    print("   - Try smaller batch sizes or models")
                return False
            # Continue to next strategy
            continue
    
    return False

def enhanced_memory_monitoring(gpu_manager: GPUManager):
    """Enhanced memory monitoring including GPU-specific metrics"""
    print(f"\n💾 Enhanced Memory Usage")
    print("=" * 50)
    
    # Standard RAM monitoring
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used / 1024**3:.1f}GB / {ram.total / 1024**3:.1f}GB ({ram.percent:.1f}%)")
    
    # GPU memory monitoring based on detection
    if gpu_manager.gpu_info and gpu_manager.gpu_info.get('is_ml_ready', False):
        gpu_type = gpu_manager.gpu_info.get('gpu_type', '').lower()
        
        if 'nvidia' in gpu_type and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                print(f"GPU Memory Allocated: {allocated / 1024**3:.1f}GB")
                print(f"GPU Memory Reserved: {reserved / 1024**3:.1f}GB")
                print(f"GPU Memory Total: {total / 1024**3:.1f}GB")
                print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
            except Exception as e:
                print(f"⚠️ NVIDIA GPU memory monitoring failed: {e}")
        
        elif 'amd' in gpu_type or 'radeon' in gpu_type:
            print(f"AMD GPU Memory: {gpu_manager.gpu_info['vram_gb']:.1f}GB total")
            if torch.cuda.is_available():
                try:
                    # Try to get basic memory info via ROCm
                    allocated = torch.cuda.memory_allocated(0)
                    print(f"GPU Memory Allocated (ROCm): {allocated / 1024**3:.1f}GB")
                except Exception as e:
                    print("💡 Install ROCm tools for detailed AMD GPU memory monitoring")
            else:
                print("💡 Install ROCm tools for detailed AMD GPU memory monitoring")
        
        else:
            print("GPU memory monitoring not available for this GPU type")
    else:
        print("No ML-capable GPU detected")

def save_model_info(tokenizer, model, device, gpu_manager, output_file="./model_info.pkl"):
    """Save model information for the RAG/Fine-tuning pipeline"""
    print(f"\n💾 Saving model info for RAG/Fine-tuning pipeline...")
    
    model_info = {
        'model': model,
        'tokenizer': tokenizer,
        'device': device,
        'gpu_info': gpu_manager.gpu_info,
        'batch_size': gpu_manager.get_optimal_batch_size(),
        'use_mixed_precision': gpu_manager.should_use_mixed_precision(),
    }
    
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"✅ Model info saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to save model info: {e}")
        return None

def launch_rag_pipeline(model_info_file, mode="both", test=True):
    """Launch the RAG/Fine-tuning pipeline with saved model info"""
    print(f"\n🚀 Launching RAG/Fine-tuning pipeline...")
    
    # Check if the RAG script exists
    rag_script = "sre_rag_finetuning.py"  # Assume you saved the RAG script as this
    
    if not Path(rag_script).exists():
        print(f"❌ RAG script not found: {rag_script}")
        print("💡 Save the RAG script as 'sre_rag_finetuning.py' in the same directory")
        return False
    
    # Build command
    cmd = [
        sys.executable, rag_script,
        "--mode", mode,
        "--model-info", model_info_file,
        "--output-dir", "./sre_outputs"
    ]
    
    if test:
        cmd.append("--test")
    
    try:
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print("✅ RAG/Fine-tuning pipeline completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Python interpreter not found: {sys.executable}")
        return False

def load_cached_model_info(model_info_file="./model_info.pkl"):
    """Load model info from cached pkl file"""
    if Path(model_info_file).exists():
        print(f"📦 Found cached model: {model_info_file}")
        try:
            with open(model_info_file, 'rb') as f:
                model_info = pickle.load(f)
            
            # Show which model was cached
            if 'model_id' in model_info:
                print(f"📝 Cached model: {model_info['model_id']}")
            else:
                print(f"⚠️ Model ID not found in cache (old cache format)")
            
            print(f"✅ Loaded cached model successfully")
            return model_info
        except Exception as e:
            print(f"⚠️ Failed to load cached model: {e}")
            print(f"🔄 Will load fresh model instead")
            return None
    else:
        print(f"📦 No cached model found at {model_info_file}")
        return None

def can_fit_model(model_size_gb: float, vram_gb: float, is_amd: bool = False) -> Tuple[bool, str]:
    """
    Check if model will fit with safety margins.
    Returns (fits, reason_string)
    """
    # Overhead for model loading, KV cache, optimizer states, etc.
    if is_amd:
        # AMD/ROCm has higher overhead
        safety_margin = 0.50  # Only use 50% of VRAM
        overhead_gb = 2.0  # Extra 2GB for ROCm runtime
    else:
        # NVIDIA
        safety_margin = 0.70  # Use 70% of VRAM
        overhead_gb = 1.5
    
    usable_vram = (vram_gb * safety_margin) - overhead_gb
    
    if model_size_gb <= usable_vram:
        return True, f"{model_size_gb:.1f}GB model fits ({usable_vram:.1f}GB available)"
    else:
        needed = model_size_gb - usable_vram
        return False, f"Model {model_size_gb:.1f}GB exceeds available {usable_vram:.1f}GB (need {needed:.1f}GB more)"

def load_model_safely(model_id: str, vram_gb: float, is_amd: bool = False):
    """
    Load a model with strict pre-flight checks.
    """
    print(f"\nPre-flight checks for {model_id}...")
    print("=" * 60)
    
    # 1. Verify model size
    print(f"1. Checking model size...")
    model_size = get_safe_model_size(model_id, verbose=True)
    
    if model_size is None:
        print("   ❌ Could not verify model size - refusing to load")
        return None, None, None
    
    # 2. Check if it fits
    print(f"2. Checking available VRAM...")
    fits, reason = can_fit_model(model_size, vram_gb, is_amd)
    print(f"   {reason}")
    
    if not fits:
        print("   ❌ Model won't fit - refusing to load")
        return None, None, None
    
    # 3. Check system RAM as buffer
    print(f"3. Checking system RAM...")
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"   Available: {available_ram_gb:.1f}GB")
    
    if available_ram_gb < 4:
        print("   ⚠️ Low system RAM - loading may still fail")
    
    # 4. Actually try to load
    print(f"4. Attempting to load model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if is_amd else torch.float16,
            device_map="auto" if not is_amd else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if is_amd and device == "cuda":
            model = model.to("cuda")
        
        print(f"   ✅ Successfully loaded on {device}")
        return tokenizer, model, device, model_id
        
    except RuntimeError as e:
        print(f"   ❌ OOM Error during loading: {e}")
        if "out of memory" in str(e).lower():
            print(f"   💡 This model doesn't fit despite predictions")
            print(f"   💡 Try a smaller model or clear GPU cache")
        return None, None, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, None, None



def get_safe_model_size(model_id: str, verbose=False) -> float:
    """
    Get model size with multiple verification methods.
    Returns None if we can't reliably determine size (fail-safe approach).
    """
    try:
        # Method 1: Try HF model card API
        response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            timeout=10
        )
        if response.ok:
            data = response.json()
            
            # Check for gated model (requires auth)
            if data.get('gated', False):
                if verbose:
                    print(f"  ⚠️ {model_id} is gated (requires authentication)")
                return None
            
            # Look at siblings (actual files)
            if data.get('siblings'):
                weights_size = 0
                for file in data.get('siblings', []):
                    fname = file.get('rfilename', '').lower()
                    # Only count model weights, not config/tokenizer
                    if any(fname.endswith(ext) for ext in ['.safetensors', '.bin', '.pt', '.pth']):
                        weights_size += file.get('size', 0)
                
                if weights_size > 0:
                    size_gb = weights_size / (1024**3)
                    if verbose:
                        print(f"  ✓ HF API: {size_gb:.2f}GB")
                    return size_gb
    except Exception as e:
        if verbose:
            print(f"  ✗ HF API failed: {e}")
    
    # Method 2: Pattern matching (fallback, but less reliable)
    patterns = {
        '405b': 810, '400b': 800,
        '175b': 350, '176b': 352,
        '70b': 140, '65b': 130, '60b': 120,
        '34b': 68, '33b': 66, '32b': 64,
        '30b': 60, '20b': 40,
        '15b': 30, '14b': 28, '13b': 26,
        '12b': 24, '11b': 22, '10b': 20,
        '8b': 16, '7b': 14, '6.7b': 13.4,
        '3b': 6, '2.7b': 5.4, '2b': 4,
        '1.5b': 3, '1.3b': 2.6, '1b': 2,
        '0.5b': 1, '500m': 1
    }
    
    model_lower = model_id.lower()
    for pattern, size_gb in sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True):
        if pattern in model_lower:
            if verbose:
                print(f"  ✓ Pattern matched: {size_gb:.2f}GB")
            return float(size_gb)
    
    # Can't determine size reliably
    if verbose:
        print(f"  ✗ Could not determine size")
    return None

def main_oom_safe():
    gpu_manager = GPUManager()
    enhanced_system_check(gpu_manager)
    
    vram_gb = gpu_manager.gpu_info.get('vram_gb', 0)
    is_amd = gpu_manager.is_amd_gpu()
    
    if vram_gb < 2:
        print("❌ Not enough VRAM for any model")
        return
    
    # Find a model that will fit
    model_id, reason = find_loadable_model(vram_gb, is_amd)
    
    if model_id is None:
        print(f"❌ {reason}")
        return
    
    # Load it safely
    tokenizer, model, device = load_model_safely(model_id, vram_gb, is_amd)
    
    if tokenizer and model:
        print("✅ Model loaded successfully!")
        # Continue with inference...
    else:
        print("❌ Failed to load model")

def main():
    """Enhanced main function with cached model detection"""
    print("🎯 Enhanced SRE AI Training - Full Pipeline (AMD GPU Compatible)")
    print("=" * 70)
    
    # Try to load cached model first
    cached_model_info = load_cached_model_info("./model_info.pkl")
    
    use_cache = True
    if cached_model_info:
        print("\nCached model found. Do you want to use it?")
        print("1. Use cached model (fast)")
        print("2. Load fresh model (skip cache)")
        
        try:
            choice = input("\nEnter choice (1-2, default 1): ").strip() or "1"
            use_cache = choice == "1"
        except (KeyboardInterrupt, EOFError):
            use_cache = True
    
    if cached_model_info and use_cache:
        print("\n✅ Using cached model")
        # Show which model is cached
        if 'model_id' in cached_model_info:
            print(f"📝 Model: {cached_model_info['model_id']}")

        tokenizer = cached_model_info['tokenizer']
        model = cached_model_info['model']
        device = cached_model_info['device']
        
        # Reconstruct GPU manager info from cache
        gpu_manager = GPUManager()
        gpu_manager.gpu_info = cached_model_info['gpu_info']
        
    else:
        # Initialize GPU manager if not using cache
        print("\n🔧 Loading fresh model...")
        gpu_manager = GPUManager()
        
        # Enhanced system check
        enhanced_system_check(gpu_manager)
        
        # Load model with GPU-optimized config
        tokenizer, model, device, model_id = load_model_with_gpu_config(gpu_manager)
        
        # Save model info for future runs
        model_info_file = save_model_info(tokenizer, model, device, gpu_manager, model_id)
    
    if tokenizer and model:
        # Run inference test (with AMD GPU error handling)
        inference_success = optimized_inference_test(tokenizer, model, device, gpu_manager)
        enhanced_memory_monitoring(gpu_manager)
        
        if inference_success:
            print(f"\n🎉 Basic setup complete!")
        else:
            print(f"\n⚠️ Basic setup complete with some issues")
            
        print(f"💡 Rust GPU detection: {gpu_manager.gpu_info['gpu_type']}")
        
        if gpu_manager.is_amd_gpu():
            print(f"🔧 AMD GPU detected - using conservative settings")
        
        # Save model info if it's fresh (for future runs)
        if not cached_model_info:
            model_info_file = save_model_info(tokenizer, model, device, gpu_manager, model_id)
        else:
            model_info_file = "./model_info.pkl"
        
        if model_info_file:
            print(f"\n🤖 Ready to launch RAG/Fine-tuning pipeline!")
            
            # Ask user what they want to do
            print("\nChoose your next step:")
            print("1. RAG system only")
            print("2. Fine-tuning only") 
            print("3. Both RAG and fine-tuning")
            print("4. Skip pipeline")
            
            try:
                choice = input("\nEnter choice (1-4): ").strip()
                
                mode_map = {
                    "1": "rag",
                    "2": "finetune", 
                    "3": "both",
                    "4": None
                }
                
                mode = mode_map.get(choice)
                
                if mode:
                    success = launch_rag_pipeline(model_info_file, mode=mode, test=True)
                    if success:
                        print(f"\n🎯 Full pipeline completed successfully!")
                        print(f"📊 Check ./sre_outputs for results")
                    else:
                        print(f"\n⚠️ Pipeline had issues, but basic model is ready")
                else:
                    print(f"\n✅ Basic setup completed - pipeline skipped")
                    
            except KeyboardInterrupt:
                print(f"\n✅ Basic setup completed - pipeline skipped")
    
    else:
        print("❌ Setup failed. Check your configuration.")

if __name__ == "__main__":
    main()
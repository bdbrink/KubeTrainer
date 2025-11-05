#!/usr/bin/env python3
"""
K8s Training Data Integration
Collects data from K8s clusters using Rust collector and prepares it for training
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import argparse


class K8sDataCollector:
    """Interface to Rust K8s data collector"""
    
    def __init__(self, rust_binary: Optional[str] = None):
        if rust_binary is None:
            # Auto-detect binary location
            possible_paths = [
                "../k8s-data-collector/target/release/k8s-data-collector",
                "./k8s-data-collector",
                "../k8s-data-collector/target/debug/k8s-data-collector"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    rust_binary = path
                    print(f"✅ Found K8s collector at: {path}")
                    break
        
        if rust_binary is None:
            raise FileNotFoundError(
                "K8s data collector binary not found. "
                "Please build it with: cd k8s-data-collector && cargo build --release"
            )
        
        self.rust_binary = rust_binary
        self.output_dir = Path("./training_data")
    
    def collect_all_data(
        self,
        namespace: Optional[str] = None,
        include_events: bool = True,
        output_format: str = "jsonl"
    ) -> Path:
        """Collect all K8s data"""
        print("\n🔍 Collecting K8s cluster data...")
        print("=" * 60)
        
        cmd = [
            self.rust_binary,
            "--output-dir", str(self.output_dir),
            "--format", output_format,
            "all"
        ]
        
        if namespace:
            cmd.extend(["--namespace", namespace])
        
        if include_events:
            cmd.append("--include-events")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            print(f"✅ Data collection complete!")
            print(f"📁 Output directory: {self.output_dir}")
            
            return self.output_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Collection failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def collect_pods_only(
        self,
        problems_only: bool = True,
        namespace: Optional[str] = None
    ) -> Path:
        """Collect only pod data"""
        print("\n🔍 Collecting pod data...")
        
        cmd = [
            self.rust_binary,
            "--output-dir", str(self.output_dir),
            "pods"
        ]
        
        if problems_only:
            cmd.append("--problems-only")
        
        if namespace:
            cmd.extend(["--namespace", namespace])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Pod data collected to {self.output_dir}")
            return self.output_dir
        except subprocess.CalledProcessError as e:
            print(f"❌ Collection failed: {e.stderr}")
            raise
    
    def generate_synthetic_data(self, count: int = 100) -> Path:
        """Generate synthetic SRE scenarios"""
        print(f"\n🎲 Generating {count} synthetic scenarios...")
        
        cmd = [
            self.rust_binary,
            "--output-dir", str(self.output_dir),
            "synthetic",
            "--count", str(count)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Generated {count} synthetic examples")
            return self.output_dir
        except subprocess.CalledProcessError as e:
            print(f"❌ Generation failed: {e.stderr}")
            raise


class TrainingDataProcessor:
    """Process collected data for model training"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load JSONL training data"""
        filepath = self.data_dir / f"{filename}.jsonl"
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            return []
        
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        return examples
    
    def combine_all_data(self) -> List[Dict]:
        """Combine all collected training data"""
        all_examples = []
        
        # Load all JSONL files in the data directory
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            print(f"📖 Loading {jsonl_file.name}...")
            
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_examples.append(json.loads(line))
        
        print(f"✅ Loaded {len(all_examples)} total examples")
        return all_examples
    
    def filter_by_severity(self, examples: List[Dict], severity: str) -> List[Dict]:
        """Filter examples by severity level"""
        return [
            ex for ex in examples
            if ex['metadata']['severity'] == severity
        ]
    
    def create_training_splits(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8
    ) -> tuple[List[Dict], List[Dict]]:
        """Split data into train/validation sets"""
        import random
        
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]
        
        return train, val
    
    def save_for_huggingface(
        self,
        examples: List[Dict],
        output_file: str = "k8s_training_dataset.jsonl"
    ):
        """Save in HuggingFace datasets format"""
        output_path = self.data_dir / output_file
        
        with open(output_path, 'w') as f:
            for example in examples:
                # Convert to instruction format
                hf_example = {
                    "instruction": example['input'],
                    "output": example['output'],
                    "metadata": example['metadata']
                }
                f.write(json.dumps(hf_example) + '\n')
        
        print(f"💾 Saved {len(examples)} examples to {output_path}")
        return output_path
    
    def print_statistics(self, examples: List[Dict]):
        """Print dataset statistics"""
        print("\n📊 Dataset Statistics")
        print("=" * 60)
        
        print(f"Total examples: {len(examples)}")
        
        # Count by resource type
        resource_types = {}
        for ex in examples:
            rt = ex['resource_type']
            resource_types[rt] = resource_types.get(rt, 0) + 1
        
        print("\nBy resource type:")
        for rt, count in sorted(resource_types.items()):
            print(f"  • {rt}: {count}")
        
        # Count by severity
        severities = {}
        for ex in examples:
            sev = ex['metadata']['severity']
            severities[sev] = severities.get(sev, 0) + 1
        
        print("\nBy severity:")
        for sev, count in sorted(severities.items()):
            print(f"  • {sev}: {count}")


def integrate_with_training_script(
    training_data_path: Path,
    model_info_file: Optional[str] = None
):
    """Integrate K8s data with the main training script"""
    print("\n🔗 Integrating with training pipeline...")
    
    # The training script can now load this data
    print(f"📁 Training data ready at: {training_data_path}")
    
    if model_info_file and Path(model_info_file).exists():
        print(f"🤖 Using model from: {model_info_file}")
        print("\n💡 You can now fine-tune with this K8s data!")
        print("   Run: python infra_learning.py")
    else:
        print("\n💡 Next steps:")
        print("   1. Run: python infra_learning.py")
        print("   2. The model will be saved for fine-tuning")
        print(f"   3. Training data is ready at: {training_data_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect K8s data for SRE AI training"
    )
    
    parser.add_argument(
        "--collector-binary",
        help="Path to Rust K8s collector binary"
    )
    
    parser.add_argument(
        "--namespace",
        help="K8s namespace to collect from (default: all)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["all", "pods", "synthetic", "events"],
        default="all",
        help="Data collection mode"
    )
    
    parser.add_argument(
        "--problems-only",
        action="store_true",
        help="Only collect problematic resources"
    )
    
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=100,
        help="Number of synthetic examples to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./training_data",
        help="Output directory for training data"
    )
    
    parser.add_argument(
        "--model-info",
        help="Path to model_info.pkl for integration"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize collector
        collector = K8sDataCollector(args.collector_binary)
        collector.output_dir = Path(args.output_dir)
        
        # Collect data based on mode
        if args.mode == "all":
            data_dir = collector.collect_all_data(
                namespace=args.namespace,
                include_events=True
            )
        elif args.mode == "pods":
            data_dir = collector.collect_pods_only(
                problems_only=args.problems_only,
                namespace=args.namespace
            )
        elif args.mode == "synthetic":
            data_dir = collector.generate_synthetic_data(
                count=args.synthetic_count
            )
        elif args.mode == "events":
            # Events collection would go here
            print("Events mode not yet implemented")
            return
        
        # Process collected data
        processor = TrainingDataProcessor(data_dir)
        all_examples = processor.combine_all_data()
        
        if all_examples:
            processor.print_statistics(all_examples)
            
            # Save in HuggingFace format
            hf_dataset = processor.save_for_huggingface(all_examples)
            
            # Integrate with training script
            integrate_with_training_script(data_dir, args.model_info)
        else:
            print("⚠️  No data collected")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 To build the K8s collector:")
        print("   cd k8s-data-collector")
        print("   cargo build --release")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
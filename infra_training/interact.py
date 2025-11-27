#!/usr/bin/env python3
"""
SRE AI Model Interaction Script
Loads cached models and provides interactive chat interface
"""

import torch
import pickle
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

class ModelInteractor:
    """Interactive chat interface for loaded models"""
    
    def __init__(self, model_info_path: str):
        """Load model from cached pickle file"""
        print("🔄 Loading cached model...")
        
        with open(model_info_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        self.tokenizer = self.model_info['tokenizer']
        self.model = self.model_info['model']
        self.device = self.model_info['device']
        self.model_id = self.model_info.get('model_id', 'Unknown Model')
        self.gpu_info = self.model_info.get('gpu_info', {})
        
        # AMD GPU detection
        self.is_amd = 'amd' in str(self.gpu_info.get('gpu_type', '')).lower()
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
        
        print(f"✅ Loaded: {self.model_id}")
        print(f"📍 Device: {self.device}")
        if self.is_amd:
            print("🔧 AMD GPU detected - using conservative settings")
        print()
    
    def _get_generation_config(self, max_tokens: int = 300) -> Dict:
        """Get generation config based on hardware"""
        config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
        }
        
        if self.is_amd and self.device == "cuda":
            # Ultra-conservative for AMD
            config.update({
                "do_sample": False,
                "num_beams": 1,
                "early_stopping": False,
                "temperature": None,
                "top_p": None,
            })
            # Set AMD-specific env vars
            os.environ['AMD_SERIALIZE_KERNEL'] = '3'
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
        else:
            # Normal sampling for NVIDIA/CPU
            config.update({
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        
        return config
    
    def generate_response(self, prompt: str, max_tokens: int = 300, use_history: bool = True) -> str:
        """Generate response from model"""
        
        # Build prompt with history if enabled
        if use_history and self.history:
            # Format conversation history
            context_parts = []
            for msg in self.history[-6:]:  # Last 3 exchanges
                context_parts.append(f"User: {msg['user']}")
                context_parts.append(f"Assistant: {msg['assistant']}")
            context_parts.append(f"User: {prompt}")
            full_prompt = "\n".join(context_parts)
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        gen_config = self._get_generation_config(max_tokens)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Strip the prompt from response
            response = response[len(full_prompt):].strip()
            
            # Store in history
            if use_history:
                self.history.append({
                    "user": prompt,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            return response
            
        except RuntimeError as e:
            if "hip" in str(e).lower() or "out of memory" in str(e).lower():
                print(f"\n⚠️ GPU error: {e}")
                print("🔧 Try reducing max_tokens or using /cpu mode")
                return "[Error: GPU issue - try /cpu or smaller responses]"
            raise e
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("💬 Interactive Chat Mode")
        print("=" * 60)
        print("Commands:")
        print("  /quit or /exit     - Exit chat")
        print("  /clear             - Clear conversation history")
        print("  /history           - Show conversation history")
        print("  /save [filename]   - Save conversation to file")
        print("  /cpu               - Switch to CPU mode (if having GPU issues)")
        print("  /tokens [N]        - Set max response tokens (default 300)")
        print("  /nohistory         - Toggle conversation history")
        print("=" * 60)
        print()
        
        use_history = True
        max_tokens = 300
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd_parts = user_input.split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else None
                    
                    if cmd in ['/quit', '/exit']:
                        print("\n👋 Goodbye!")
                        break
                    
                    elif cmd == '/clear':
                        self.history.clear()
                        print("🗑️  Conversation history cleared")
                        continue
                    
                    elif cmd == '/history':
                        if not self.history:
                            print("📝 No conversation history")
                        else:
                            print(f"\n📝 Conversation History ({len(self.history)} exchanges)")
                            print("-" * 60)
                            for i, msg in enumerate(self.history, 1):
                                print(f"\n[{i}] {msg.get('timestamp', 'N/A')}")
                                print(f"You: {msg['user'][:80]}...")
                                print(f"AI:  {msg['assistant'][:80]}...")
                            print()
                        continue
                    
                    elif cmd == '/save':
                        filename = arg or f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        self._save_conversation(filename)
                        continue
                    
                    elif cmd == '/cpu':
                        if self.device == "cuda":
                            print("🔄 Moving model to CPU...")
                            self.model = self.model.to('cpu')
                            self.device = 'cpu'
                            print("✅ Now running on CPU")
                        else:
                            print("ℹ️  Already on CPU")
                        continue
                    
                    elif cmd == '/tokens':
                        if arg and arg.isdigit():
                            max_tokens = int(arg)
                            print(f"✅ Max tokens set to {max_tokens}")
                        else:
                            print(f"ℹ️  Current max tokens: {max_tokens}")
                        continue
                    
                    elif cmd == '/nohistory':
                        use_history = not use_history
                        print(f"✅ Conversation history: {'ON' if use_history else 'OFF'}")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {cmd}")
                        continue
                
                # Generate response
                print("AI: ", end="", flush=True)
                response = self.generate_response(user_input, max_tokens, use_history)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Try /cpu if having GPU issues, or /quit to exit\n")
    
    def _save_conversation(self, filename: str):
        """Save conversation history to file"""
        try:
            with open(filename, 'w') as f:
                f.write(f"Conversation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_id}\n")
                f.write(f"Device: {self.device}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, msg in enumerate(self.history, 1):
                    f.write(f"[Exchange {i}] {msg.get('timestamp', 'N/A')}\n")
                    f.write(f"You: {msg['user']}\n\n")
                    f.write(f"AI: {msg['assistant']}\n")
                    f.write("-" * 60 + "\n\n")
            
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save: {e}")

def find_cached_models(models_dir: str = "./models") -> List[Path]:
    """Find all cached model pickle files"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return []
    
    pkl_files = list(models_path.glob("*/model_info.pkl"))
    
    # Also check root
    root_pkl = models_path / "model_info.pkl"
    if root_pkl.exists():
        pkl_files.append(root_pkl)
    
    return pkl_files

def select_model() -> Optional[Path]:
    """Interactive model selection"""
    pkl_files = find_cached_models()
    
    if not pkl_files:
        print("❌ No cached models found in ./models directory")
        print("💡 Run sre_training.py first to set up a model")
        return None
    
    if len(pkl_files) == 1:
        print(f"📦 Found 1 cached model: {pkl_files[0].parent.name}")
        return pkl_files[0]
    
    print(f"\n📦 Found {len(pkl_files)} cached models:\n")
    
    for i, pkl_file in enumerate(pkl_files, 1):
        model_name = pkl_file.parent.name if pkl_file.parent.name != "models" else "root"
        mod_time = pkl_file.stat().st_mtime
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i}. {model_name} (modified: {mod_date})")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(pkl_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(pkl_files):
                    return pkl_files[idx]
            
            print("❌ Invalid choice")
        except (KeyboardInterrupt, EOFError):
            return None

def main():
    """Main entry point"""
    print("🤖 SRE AI Model Interaction")
    print("=" * 60)
    print()
    
    # Select model
    model_path = select_model()
    
    if not model_path:
        print("❌ No model selected. Exiting.")
        return
    
    print()
    
    # Load and start chat
    try:
        interactor = ModelInteractor(str(model_path))
        interactor.chat_loop()
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
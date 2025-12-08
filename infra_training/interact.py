def _ask_permission(self, command: str) -> bool:
        """Ask user for permission to run command"""
        if self.auto_approve:
            return True
        
        print(f"\n🤖 Model wants to run: {command}")
        while True:
            response = input("   Allow? [y/n/always]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['a', 'always']:
                self.auto_approve = True
                print("   ✅ Auto-approve enabled for this session")
                return True
            else:
                print("   Please enter y, n, or always")#!/usr/bin/env python3
"""
SRE AI Model Interaction Script with Code Viewing and Command Execution
Loads cached models and provides interactive chat interface with file system access and kubectl/shell commands
"""

import torch
import pickle
import os
import sys
import warnings
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

class CommandExecutor:
    """Safely executes shell commands with allowlist"""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None):
        # Default safe commands for SRE work
        self.allowed_commands = allowed_commands or [
            'kubectl', 'docker', 'helm', 'git', 'ls', 'cat', 'grep', 
            'ps', 'df', 'du', 'top', 'netstat', 'curl', 'ping',
            'systemctl', 'journalctl', 'free', 'uptime', 'whoami',
            'aws', 'gcloud', 'az'  # Cloud CLIs
        ]
        self.timeout = 30  # seconds
    
    def is_allowed(self, command: str) -> Tuple[bool, str]:
        """Check if command is in allowlist"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        
        # Check if base command is allowed
        if base_cmd not in self.allowed_commands:
            return False, f"Command '{base_cmd}' not in allowlist"
        
        # Block dangerous patterns
        dangerous = ['rm', 'delete', 'drop', 'truncate', '>', '>>', 'sudo', 'su']
        for danger in dangerous:
            if danger in command.lower():
                return False, f"Dangerous operation detected: {danger}"
        
        return True, "Allowed"
    
    def execute(self, command: str) -> Dict[str, any]:
        """Execute command and return output"""
        allowed, reason = self.is_allowed(command)
        
        if not allowed:
            return {
                'success': False,
                'error': reason,
                'stdout': '',
                'stderr': reason
            }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Command timed out after {self.timeout}s',
                'stdout': '',
                'stderr': 'Timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': str(e)
            }

class CodeContext:
    """Manages code file access and context injection"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.allowed_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
                                   '.go', '.rs', '.rb', '.php', '.sh', '.yaml', '.yml',
                                   '.json', '.xml', '.md', '.txt', '.csv', '.sql', '.tf'}
        self.max_file_size = 100_000  # 100KB max per file
    
    def list_files(self, directory: str = ".", pattern: str = "*.py") -> List[Path]:
        """List files in directory matching pattern"""
        try:
            search_path = (self.root_dir / directory).resolve()
            
            # Security: ensure we stay within root
            if not str(search_path).startswith(str(self.root_dir)):
                return []
            
            if not search_path.exists():
                return []
            
            files = []
            for item in search_path.glob(pattern):
                if item.is_file() and item.suffix in self.allowed_extensions:
                    rel_path = item.relative_to(self.root_dir)
                    files.append(rel_path)
            
            return sorted(files)
        except Exception:
            return []
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Safely read a file and return its contents"""
        try:
            full_path = (self.root_dir / filepath).resolve()
            
            # Security checks
            if not str(full_path).startswith(str(self.root_dir)):
                return None
            
            if not full_path.exists() or not full_path.is_file():
                return None
            
            if full_path.suffix not in self.allowed_extensions:
                return None
            
            if full_path.stat().st_size > self.max_file_size:
                return f"[File too large: {full_path.stat().st_size / 1024:.1f}KB]"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"[Error reading file: {e}]"

class ModelInteractor:
    """Interactive chat interface for loaded models with code context"""
    
    def __init__(self, model_info_path: str, enable_commands: bool = True, enable_files: bool = True, auto_approve: bool = False):
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
        
        # Set AMD-specific env vars early
        if self.is_amd and self.device == "cuda":
            os.environ['AMD_SERIALIZE_KERNEL'] = '3'
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
            os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '0'
        
        # Context managers
        self.enable_commands = enable_commands
        self.enable_files = enable_files
        self.auto_approve = auto_approve  # Auto-approve commands or ask?
        self.command_executor = CommandExecutor() if enable_commands else None
        self.code_context = CodeContext() if enable_files else None
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
        
        # Session stats
        self.session_start = datetime.now()
        self.total_tokens_generated = 0
        self.total_tokens_input = 0
        self.total_generation_time = 0.0
        self.message_count = 0
        
        print(f"✅ Loaded: {self.model_id}")
        print(f"📍 Device: {self.device}")
        if self.is_amd:
            print("🔧 AMD GPU detected - using conservative settings")
        if self.enable_commands:
            mode = "auto-approve" if auto_approve else "ask permission"
            print(f"⚡ Command execution: ENABLED ({mode})")
        if self.enable_files:
            print("📁 File access: ENABLED")
        print()
    
    def _ask_permission(self, command: str) -> bool:
        """Ask user for permission to run command"""
        if self.auto_approve:
            return True
        
        print(f"\n🤖 Model wants to run: {command}")
        while True:
            try:
                response = input("   Allow? [y/n/always]: ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['a', 'always']:
                    self.auto_approve = True
                    print("   ✅ Auto-approve enabled for this session")
                    return True
                else:
                    print("   Please enter y, n, or always")
            except (KeyboardInterrupt, EOFError):
                print("\n   ⛔ Denied")
                return False
    
    def _inject_context(self, user_prompt: str) -> Tuple[str, str]:
        """Inject system context and available tools into prompt"""
        context_parts = []
        tool_info = []
        
        # System context
        context_parts.append(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_parts.append(f"Working directory: {os.getcwd()}")
        
        # Available tools with better instructions
        if self.enable_commands:
            tool_info.append("TOOL: Execute commands with [EXEC:command] (use ONCE per command)")
            tool_info.append(f"Available: {', '.join(self.command_executor.allowed_commands[:8])}")
        
        if self.enable_files:
            tool_info.append("TOOL: Read files with [READ:filepath]")
            tool_info.append("TOOL: List files with [LIST:dir pattern]")
        
        system_context = "\n".join(context_parts)
        tools_context = "\n".join(tool_info) if tool_info else ""
        
        # More direct prompt format with examples
        enhanced_prompt = f"""You are an SRE assistant. You can propose commands to run.

{tools_context}

Context: {system_context}

User asks: {user_prompt}

To propose a command, write: [EXEC:command]
Example: [EXEC:kubectl get nodes]

You will be asked for permission before the command runs.
After seeing the output, analyze ONLY the actual data shown (don't make up examples).
Be brief (2-3 sentences).

Your response:"""
        
        return enhanced_prompt, system_context
    
    def _process_commands(self, text: str) -> str:
        """Process any embedded commands in the text"""
        if not self.enable_commands:
            return text
        
        import re
        
        # Look for [EXEC:...] patterns (primary format)
        exec_pattern = r'\[EXEC:\s*([^\]]+?)\]'
        
        # Also look for plain "EXEC: command" without brackets (fallback)
        plain_exec_pattern = r'(?:^|\n)EXEC:\s*([^\n]+)'
        
        # Find all unique commands from both patterns
        commands_found = re.findall(exec_pattern, text)
        commands_found.extend(re.findall(plain_exec_pattern, text))
        
        unique_commands = []
        seen = set()
        for cmd in commands_found:
            cmd_clean = cmd.strip()
            if cmd_clean not in seen:
                unique_commands.append(cmd_clean)
                seen.add(cmd_clean)
        
        # Execute each unique command with permission
        results = {}
        for command in unique_commands:
            # Ask for permission
            if not self._ask_permission(command):
                results[command] = f"\n⛔ Command denied by user: {command}\n"
                continue
            
            print(f"🔧 Executing: {command}")
            result = self.command_executor.execute(command)
            
            if result['success']:
                output = result['stdout'].strip()
                if len(output) > 1500:
                    output = output[:1500] + "\n... (truncated)"
                results[command] = f"\n```bash\n$ {command}\n{output}\n```\n"
            else:
                stderr = result.get('stderr', 'Unknown error')
                # Truncate long error messages
                if len(stderr) > 500:
                    stderr = stderr[:500] + "..."
                results[command] = f"\n```bash\n$ {command}\nError: {stderr}\n```\n"
        
        # Replace all occurrences with results
        def replace_exec(match):
            cmd = match.group(1).strip()
            return results.get(cmd, f"[Command: {cmd}]")
        
        # Replace both patterns
        text = re.sub(exec_pattern, replace_exec, text)
        text = re.sub(plain_exec_pattern, replace_exec, text)
        
        return text
    
    def _process_file_reads(self, text: str) -> str:
        """Process any file read requests"""
        if not self.enable_files:
            return text
        
        import re
        
        # [READ: filepath]
        read_pattern = r'\[READ:\s*([^\]]+)\]'
        
        def read_and_replace(match):
            filepath = match.group(1).strip()
            print(f"\n📄 Reading: {filepath}")
            content = self.code_context.read_file(filepath)
            
            if content:
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                return f"\n```\n{content}\n```\n"
            else:
                return f"\n```\nError: Could not read {filepath}\n```\n"
        
        # [LIST: pattern]
        list_pattern = r'\[LIST:\s*([^\]]+)\]'
        
        def list_and_replace(match):
            pattern = match.group(1).strip()
            parts = pattern.split(maxsplit=1)
            directory = parts[0] if parts else "."
            file_pattern = parts[1] if len(parts) > 1 else "*.py"
            
            print(f"\n📁 Listing: {directory}/{file_pattern}")
            files = self.code_context.list_files(directory, file_pattern)
            
            if files:
                file_list = "\n".join(f"  - {f}" for f in files[:50])
                if len(files) > 50:
                    file_list += f"\n  ... and {len(files) - 50} more"
                return f"\n```\nFiles:\n{file_list}\n```\n"
            else:
                return f"\n```\nNo files found matching {directory}/{file_pattern}\n```\n"
        
        text = re.sub(read_pattern, read_and_replace, text)
        text = re.sub(list_pattern, list_and_replace, text)
        
        return text
    
    def _get_generation_config(self, max_tokens: int = 300) -> Dict:
        """Get generation config based on hardware"""
        config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": False,
        }
        
        if self.is_amd and self.device == "cuda":
            config.update({
                "do_sample": False,
                "num_beams": 1,
            })
        else:
            config.update({
                "do_sample": True,
                "temperature": 0.5,  # Lower temp = less creative/hallucination
                "top_p": 0.85,        # Tighter sampling
                "top_k": 40,
                "repetition_penalty": 1.3,  # Stronger penalty
            })
        
        return config
    
    def generate_response(self, prompt: str, max_tokens: int = 300, use_history: bool = True) -> str:
        """Generate response from model with context injection"""
        
        # Inject context and tools
        enhanced_prompt, sys_context = self._inject_context(prompt)
        
        # Build prompt with history if enabled
        if use_history and self.history:
            context_parts = []
            for msg in self.history[-2:]:  # Only last 2 exchanges
                context_parts.append(f"User: {msg['user']}")
                # Truncate long assistant responses in history
                resp = msg['assistant']
                if len(resp) > 150:
                    resp = resp[:150] + "..."
                context_parts.append(f"Assistant: {resp}")
            context_parts.append(enhanced_prompt)
            full_prompt = "\n\n".join(context_parts)
        else:
            full_prompt = enhanced_prompt
        
        # Tokenize with aggressive truncation
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1200)
        input_length = inputs['input_ids'].shape[1]
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with stricter settings
        gen_config = self._get_generation_config(max_tokens)
        
        try:
            import time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            generation_time = time.time() - start_time
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate tokens generated
            output_length = outputs.shape[1]
            tokens_generated = output_length - input_length
            
            # Update stats
            self.total_tokens_input += input_length
            self.total_tokens_generated += tokens_generated
            self.total_generation_time += generation_time
            self.message_count += 1
            
            # Strip the prompt from response
            response = response[len(full_prompt):].strip()
            
            # FIRST: Process embedded commands to inject real output
            response = self._process_commands(response)
            response = self._process_file_reads(response)
            
            # THEN: Aggressively stop at repetition/hallucination markers
            stop_markers = [
                '\n\nUser:', '\n\n---', '\n\nQ:', 'In summary',
                '\nHere is', '\nFor example', '\nThis output shows',
                '```\n```',  # Double code blocks = hallucination
                'The output from',  # Often precedes made-up examples
            ]
            
            earliest_stop = len(response)
            for marker in stop_markers:
                idx = response.find(marker)
                if idx > 50 and idx < earliest_stop:  # Must be after first 50 chars
                    earliest_stop = idx
            
            response = response[:earliest_stop].strip()
            
            # Remove duplicate consecutive lines (repetition detection)
            lines = response.split('\n')
            deduped = []
            prev_line = None
            for line in lines:
                if line.strip() != prev_line:
                    deduped.append(line)
                    prev_line = line.strip()
            response = '\n'.join(deduped)
            
            # Store in history
            if use_history:
                self.history.append({
                    "user": prompt,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat(),
                    "tokens_in": input_length,
                    "tokens_out": tokens_generated,
                    "generation_time": generation_time
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
        print("💬 Interactive Chat Mode with SRE Tools")
        print("=" * 60)
        print("Commands:")
        print("  /quit or /exit     - Exit chat")
        print("  /clear             - Clear conversation history")
        print("  /history           - Show conversation history")
        print("  /stats             - Show session statistics")
        print("  /exec <command>    - Execute shell command directly")
        print("  /read <file>       - Read file contents directly")
        print("  /list <dir> <pat>  - List files directly")
        print("  /kubectl <args>    - Run kubectl command")
        print("  /save [filename]   - Save conversation to file")
        print("  /cpu               - Switch to CPU mode")
        print("  /tokens [N]        - Set max response tokens (default 500)")
        print("=" * 60)
        print()
        
        if self.enable_commands:
            print("💡 The model can execute commands when it sees: [EXEC: command]")
        if self.enable_files:
            print("💡 The model can read files when it sees: [READ: filepath]")
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
                        self._print_session_summary()
                        print("\n👋 Goodbye!")
                        break
                    
                    elif cmd == '/clear':
                        self.history.clear()
                        print("🗑️  Conversation history cleared")
                        continue
                    
                    elif cmd == '/stats':
                        self._print_session_summary()
                        continue
                    
                    elif cmd == '/exec' and arg:
                        if self.command_executor:
                            print(f"⚡ Executing: {arg}")
                            result = self.command_executor.execute(arg)
                            if result['success']:
                                print(f"✅ Output:\n{result['stdout']}")
                            else:
                                print(f"❌ Error:\n{result['stderr']}")
                        else:
                            print("❌ Command execution disabled")
                        continue
                    
                    elif cmd == '/kubectl' and arg:
                        if self.command_executor:
                            kubectl_cmd = f"kubectl {arg}"
                            print(f"⚡ Executing: {kubectl_cmd}")
                            result = self.command_executor.execute(kubectl_cmd)
                            if result['success']:
                                print(f"✅ Output:\n{result['stdout']}")
                            else:
                                print(f"❌ Error:\n{result['stderr']}")
                        else:
                            print("❌ Command execution disabled")
                        continue
                    
                    elif cmd == '/read' and arg:
                        if self.code_context:
                            content = self.code_context.read_file(arg)
                            if content:
                                print(f"📄 {arg}:\n{content}")
                            else:
                                print(f"❌ Could not read {arg}")
                        else:
                            print("❌ File access disabled")
                        continue
                    
                    elif cmd == '/list':
                        if self.code_context:
                            parts = arg.split() if arg else ["."]
                            directory = parts[0] if parts else "."
                            pattern = parts[1] if len(parts) > 1 else "*.py"
                            files = self.code_context.list_files(directory, pattern)
                            if files:
                                print(f"📁 Files in {directory} matching {pattern}:")
                                for f in files:
                                    print(f"  - {f}")
                            else:
                                print(f"❌ No files found")
                        else:
                            print("❌ File access disabled")
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
                                print(f"Assistant: {msg['assistant'][:80]}...")
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
                    
                    elif cmd == '/auto':
                        self.auto_approve = not self.auto_approve
                        status = "ON" if self.auto_approve else "OFF"
                        print(f"✅ Auto-approve commands: {status}")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {cmd}")
                        continue
                
                # Generate response
                print("kubepilot: ", end="", flush=True)
                response = self.generate_response(user_input, max_tokens, use_history)
                print(response)
                print()
                
            except KeyboardInterrupt:
                self._print_session_summary()
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                self._print_session_summary()
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
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
                    f.write(f"Assistant: {msg['assistant']}\n")
                    f.write("-" * 60 + "\n\n")
            
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save: {e}")
    
    def _print_session_summary(self):
        """Print session statistics summary"""
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        print(f"\n⏱️  Session Duration: {self._format_duration(session_duration)}")
        print(f"💬 Messages: {self.message_count}")
        print(f"📊 Tokens: {self.total_tokens_input + self.total_tokens_generated:,}")
        
        if self.total_generation_time > 0:
            tokens_per_sec = self.total_tokens_generated / self.total_generation_time
            print(f"🚀 Speed: {tokens_per_sec:.1f} tokens/sec")
        
        print(f"🖥️  Device: {self.device}")
        print("=" * 60)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h {mins}m"

def find_cached_models(models_dir: str = "./models") -> List[Path]:
    """Find all cached model pickle files"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return []
    
    pkl_files = list(models_path.glob("*/model_info.pkl"))
    
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
    print("🤖 SRE AI Model Interaction with Tool Access")
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
        interactor = ModelInteractor(
            str(model_path),
            enable_commands=True,   # Set to False to disable shell commands
            enable_files=True,      # Set to False to disable file access
            auto_approve=False      # Set to True to skip permission prompts
        )
        interactor.chat_loop()
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
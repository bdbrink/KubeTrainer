#!/usr/bin/env python3
"""
Enhanced SRE AI Model Interaction - Cursor/Claude-like Experience
Features: streaming output, automatic tool use, rich UI, smart context
"""

import torch
import pickle
import os
import sys
import warnings
import subprocess
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Generator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Rich terminal output (optional, graceful fallback)
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.live import Live
    from rich.spinner import Spinner
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("💡 Install 'rich' for better formatting: pip install rich")

class ToolType(Enum):
    """Available tool types"""
    EXEC = "exec"
    READ = "read"
    LIST = "list"
    KUBECTL = "kubectl"

@dataclass
class ToolCall:
    """Represents a tool invocation"""
    tool_type: ToolType
    arguments: str
    result: Optional[str] = None
    success: bool = False

class CommandExecutor:
    """Safely executes shell commands"""
    
    def __init__(self):
        self.allowed_commands = [
            'kubectl', 'docker', 'helm', 'git', 'ls', 'cat', 'grep', 
            'ps', 'df', 'du', 'top', 'netstat', 'curl', 'ping',
            'systemctl', 'journalctl', 'free', 'uptime', 'whoami',
            'aws', 'gcloud', 'az', 'jq', 'yq', 'echo', 'pwd'
        ]
        self.timeout = 30
    
    def is_allowed(self, command: str) -> Tuple[bool, str]:
        """Check if command is allowed"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        
        if base_cmd not in self.allowed_commands:
            return False, f"Command '{base_cmd}' not in allowlist"
        
        dangerous = ['rm -rf', 'delete', 'drop', 'truncate', 'mkfs', 'dd if=', '> /dev']
        for danger in dangerous:
            if danger in command.lower():
                return False, f"Dangerous operation: {danger}"
        
        return True, "Allowed"
    
    def execute(self, command: str) -> Dict:
        """Execute command and return structured result"""
        allowed, reason = self.is_allowed(command)
        
        if not allowed:
            return {
                'success': False,
                'stdout': '',
                'stderr': reason,
                'command': command
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
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Timeout after {self.timeout}s',
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'command': command
            }

class CodeContext:
    """File system access"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.allowed_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
            '.go', '.rs', '.rb', '.php', '.sh', '.yaml', '.yml',
            '.json', '.xml', '.md', '.txt', '.csv', '.sql', '.tf',
            '.hcl', '.toml', '.ini', '.conf', '.log'
        }
        self.max_file_size = 200_000
    
    def list_files(self, directory: str = ".", pattern: str = "*") -> List[str]:
        """List files matching pattern"""
        try:
            search_path = (self.root_dir / directory).resolve()
            
            if not str(search_path).startswith(str(self.root_dir)):
                return []
            
            if not search_path.exists():
                return []
            
            files = []
            for item in search_path.rglob(pattern):
                if item.is_file() and item.suffix in self.allowed_extensions:
                    try:
                        rel_path = str(item.relative_to(self.root_dir))
                        files.append(rel_path)
                    except ValueError:
                        continue
            
            return sorted(files)[:100]  # Limit results
        except Exception:
            return []
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Read file contents"""
        try:
            full_path = (self.root_dir / filepath).resolve()
            
            if not str(full_path).startswith(str(self.root_dir)):
                return None
            
            if not full_path.exists() or not full_path.is_file():
                return None
            
            if full_path.suffix not in self.allowed_extensions:
                return None
            
            size = full_path.stat().st_size
            if size > self.max_file_size:
                return f"[File too large: {size / 1024:.1f}KB]"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"[Error: {e}]"

class StreamingGenerator:
    """Handles streaming token generation"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_stream(self, inputs: Dict, max_tokens: int = 512) -> Generator[str, None, None]:
        """Generate tokens one at a time (streaming simulation)"""
        # For true streaming, we'd need to modify the generation loop
        # This is a simplified version that yields chunks
        
        gen_config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        
        with torch.no_grad():
            # Generate full output (in production, use generation callbacks for true streaming)
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode and yield in chunks to simulate streaming
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        input_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        response = full_text[len(input_text):].strip()
        
        # Yield in word-sized chunks for smooth streaming effect
        words = response.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= 3:  # Yield every 3 words
                yield " ".join(current_chunk) + " "
                current_chunk = []
        
        if current_chunk:
            yield " ".join(current_chunk)

class SmartModelInteractor:
    """Enhanced model interaction with Cursor/Claude-like experience"""
    
    def __init__(self, model_path: str):
        """Initialize with cached model"""
        
        if RICH_AVAILABLE:
            self.console = Console()
            self.console.print("[bold cyan]🔄 Loading model...[/bold cyan]")
        else:
            print("🔄 Loading model...")
        
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.tokenizer = model_info['tokenizer']
        self.model = model_info['model']
        self.device = model_info['device']
        self.model_id = model_info.get('model_id', 'Unknown')
        
        # Tools
        self.executor = CommandExecutor()
        self.code_context = CodeContext()
        self.streamer = StreamingGenerator(self.model, self.tokenizer, self.device)
        
        # Conversation state
        self.messages = []  # Chat history in standard format
        self.max_history = 10  # Keep last 10 exchanges
        
        if RICH_AVAILABLE:
            self.console.print(f"[bold green]✅ Loaded: {self.model_id}[/bold green]")
            self.console.print(f"[dim]Device: {self.device}[/dim]\n")
        else:
            print(f"✅ Loaded: {self.model_id}")
            print(f"Device: {self.device}\n")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions"""
        return """You are an expert SRE assistant with access to system tools. You help debug, monitor, and manage infrastructure.

Available Tools (use when needed):
- [EXEC: command] - Execute shell commands (kubectl, docker, etc.)
- [READ: filepath] - Read file contents
- [LIST: directory pattern] - List files (e.g., "LIST: . *.yaml")

Guidelines:
1. Be concise and practical
2. Use tools when you need real data - don't make up examples
3. Provide actionable insights
4. Format code/commands in markdown
5. One tool call at a time for clarity

Current context:
- Time: {time}
- Directory: {cwd}
"""
    
    def _format_messages_for_model(self, user_input: str) -> str:
        """Convert chat history to model prompt format"""
        
        system_prompt = self._build_system_prompt().format(
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            cwd=os.getcwd()
        )
        
        # Build conversation
        parts = [system_prompt]
        
        # Add recent history (last few exchanges)
        for msg in self.messages[-(self.max_history*2):]:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                parts.append(f"\nUser: {content}")
            else:
                parts.append(f"\nAssistant: {content}")
        
        # Add current input
        parts.append(f"\nUser: {user_input}\n\nAssistant:")
        
        return "\n".join(parts)
    
    def _extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from model output"""
        tools = []
        
        # Match tool patterns - both [TOOL: arg] and plain TOOL: arg formats
        patterns = [
            (r'\[EXEC:\s*([^\]]+)\]', ToolType.EXEC),
            (r'\[READ:\s*([^\]]+)\]', ToolType.READ),
            (r'\[LIST:\s*([^\]]+)\]', ToolType.LIST),
            # Also catch plain format without brackets (model sometimes does this)
            (r'(?:^|\n)EXEC:\s*([^\n]+)', ToolType.EXEC),
            (r'(?:^|\n)READ:\s*([^\n]+)', ToolType.READ),
            (r'(?:^|\n)LIST:\s*([^\n]+)', ToolType.LIST),
        ]
        
        seen = set()  # Avoid duplicates
        
        for pattern, tool_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                arg = match.group(1).strip()
                # Remove any trailing punctuation/noise
                arg = re.sub(r'[`\'"]+
    
    def _execute_tool(self, tool: ToolCall) -> str:
        """Execute a tool and return formatted result"""
        
        if tool.tool_type == ToolType.EXEC:
            result = self.executor.execute(tool.arguments)
            if result['success']:
                output = result['stdout'][:2000]  # Truncate long output
                return f"\n```bash\n$ {tool.arguments}\n{output}\n```\n"
            else:
                error = result['stderr'][:500]
                return f"\n```bash\n$ {tool.arguments}\n❌ Error: {error}\n```\n"
        
        elif tool.tool_type == ToolType.READ:
            content = self.code_context.read_file(tool.arguments)
            if content and not content.startswith('['):
                # Detect language for syntax highlighting
                ext = Path(tool.arguments).suffix
                lang_map = {'.py': 'python', '.js': 'javascript', '.yaml': 'yaml', 
                           '.json': 'json', '.sh': 'bash', '.md': 'markdown'}
                lang = lang_map.get(ext, 'text')
                
                content = content[:2000]  # Truncate
                return f"\n```{lang}\n# {tool.arguments}\n{content}\n```\n"
            else:
                return f"\n❌ Could not read: {tool.arguments}\n"
        
        elif tool.tool_type == ToolType.LIST:
            parts = tool.arguments.split(maxsplit=1)
            directory = parts[0] if parts else "."
            pattern = parts[1] if len(parts) > 1 else "*"
            
            # Handle patterns like "./*.txt" -> directory=".", pattern="*.txt"
            if '/' in directory:
                path_parts = directory.rsplit('/', 1)
                directory = path_parts[0] or '.'
                pattern = path_parts[1] if path_parts[1] else pattern
            
            files = self.code_context.list_files(directory, pattern)
            if files:
                file_list = "\n".join(f"  • {f}" for f in files[:30])
                if len(files) > 30:
                    file_list += f"\n  ... and {len(files) - 30} more"
                return f"\n**Files matching {directory}/{pattern}:**\n{file_list}\n"
            else:
                return f"\n❌ No files found: {directory}/{pattern}\n"
        
        return ""
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """Process user input and generate response with streaming"""
        
        # Add to history
        self.messages.append({"role": "user", "content": user_input})
        
        # Build prompt
        prompt = self._format_messages_for_model(user_input)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with streaming
        response_parts = []
        
        if stream and RICH_AVAILABLE:
            # Rich streaming display
            with Live("", console=self.console, refresh_per_second=10) as live:
                for chunk in self.streamer.generate_stream(inputs, max_tokens=512):
                    response_parts.append(chunk)
                    current_text = "".join(response_parts)
                    
                    # Render markdown in real-time
                    try:
                        md = Markdown(current_text)
                        live.update(md)
                    except:
                        live.update(current_text)
        else:
            # Non-streaming fallback
            for chunk in self.streamer.generate_stream(inputs, max_tokens=512):
                response_parts.append(chunk)
                if not stream:
                    print(chunk, end="", flush=True)
        
        response = "".join(response_parts).strip()
        
        # Process tool calls
        tool_calls = self._extract_tool_calls(response)
        
        if tool_calls:
            if RICH_AVAILABLE:
                self.console.print("\n[dim]Executing tools...[/dim]")
            else:
                print("\n🔧 Executing tools...")
            
            # Execute tools and inject results
            for tool in tool_calls:
                if RICH_AVAILABLE:
                    self.console.print(f"[cyan]→ {tool.tool_type.value}: {tool.arguments}[/cyan]")
                else:
                    print(f"→ {tool.tool_type.value}: {tool.arguments}")
                
                tool_result = self._execute_tool(tool)
                
                # Replace tool call with result in response
                # Handle both bracketed and plain formats
                patterns_to_replace = [
                    re.escape(f"[{tool.tool_type.value.upper()}: {tool.arguments}]"),
                    re.escape(f"[{tool.tool_type.value.upper()}:{tool.arguments}]"),  # No space
                    re.escape(f"{tool.tool_type.value.upper()}: {tool.arguments}"),
                    re.escape(f"{tool.tool_type.value.upper()}:{tool.arguments}"),
                ]
                
                for pattern in patterns_to_replace:
                    response = re.sub(pattern, tool_result, response, flags=re.IGNORECASE)
        
        # Add to history
        self.messages.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
        
        return response
    
    def run_interactive(self):
        """Run interactive chat loop"""
        
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "[bold cyan]SRE AI Assistant[/bold cyan]\n"
                "Commands: /help, /clear, /quit, /exec <cmd>, /read <file>",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print("SRE AI Assistant")
            print("Commands: /help, /clear, /quit, /exec <cmd>, /read <file>")
            print("=" * 60)
        
        while True:
            try:
                # Get input
                if RICH_AVAILABLE:
                    user_input = self.console.input("\n[bold green]You:[/bold green] ").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit', '/q']:
                        break
                    
                    elif user_input == '/clear':
                        self.messages = []
                        if RICH_AVAILABLE:
                            self.console.clear()
                        print("🗑️  Cleared history")
                        continue
                    
                    elif user_input == '/help':
                        help_text = """
**Commands:**
• `/quit` - Exit
• `/clear` - Clear history  
• `/exec <command>` - Run command directly
• `/read <file>` - Read file directly

**Usage:**
Just chat naturally! The assistant will use tools when needed.
Examples:
• "Show me all pods in the default namespace"
• "Read the config.yaml file"
• "What's the CPU usage?"
"""
                        if RICH_AVAILABLE:
                            self.console.print(Markdown(help_text))
                        else:
                            print(help_text)
                        continue
                    
                    elif user_input.startswith('/exec '):
                        cmd = user_input[6:]
                        result = self.executor.execute(cmd)
                        if result['success']:
                            print(f"```\n{result['stdout']}\n```")
                        else:
                            print(f"Error: {result['stderr']}")
                        continue
                    
                    elif user_input.startswith('/read '):
                        filepath = user_input[6:]
                        content = self.code_context.read_file(filepath)
                        if content:
                            print(f"```\n{content}\n```")
                        else:
                            print(f"❌ Could not read {filepath}")
                        continue
                
                # Generate response
                if RICH_AVAILABLE:
                    self.console.print("\n[bold cyan]Assistant:[/bold cyan]", end=" ")
                else:
                    print("\nAssistant: ", end="")
                
                response = self.chat(user_input, stream=True)
                
                # Display final response if not already shown via streaming
                if not RICH_AVAILABLE or not response:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Entry point"""
    
    # Find models
    models_dir = Path("./models")
    pkl_files = []
    
    if models_dir.exists():
        pkl_files = list(models_dir.glob("*/model_info.pkl"))
        root_pkl = models_dir / "model_info.pkl"
        if root_pkl.exists():
            pkl_files.append(root_pkl)
    
    if not pkl_files:
        print("❌ No cached models found")
        print("💡 Run your training script first")
        return
    
    # Select model
    if len(pkl_files) == 1:
        model_path = pkl_files[0]
    else:
        print("\n📦 Available models:\n")
        for i, p in enumerate(pkl_files, 1):
            print(f"  {i}. {p.parent.name}")
        
        choice = input(f"\nSelect (1-{len(pkl_files)}): ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(pkl_files):
            print("❌ Invalid choice")
            return
        
        model_path = pkl_files[int(choice) - 1]
    
    # Run
    try:
        interactor = SmartModelInteractor(str(model_path))
        interactor.run_interactive()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
, '', arg)
                
                key = (tool_type, arg)
                if key not in seen:
                    tools.append(ToolCall(tool_type, arg))
                    seen.add(key)
        
        return tools
    
    def _execute_tool(self, tool: ToolCall) -> str:
        """Execute a tool and return formatted result"""
        
        if tool.tool_type == ToolType.EXEC:
            result = self.executor.execute(tool.arguments)
            if result['success']:
                output = result['stdout'][:2000]  # Truncate long output
                return f"```bash\n$ {tool.arguments}\n{output}\n```"
            else:
                return f"```bash\n$ {tool.arguments}\nError: {result['stderr'][:500]}\n```"
        
        elif tool.tool_type == ToolType.READ:
            content = self.code_context.read_file(tool.arguments)
            if content:
                # Detect language for syntax highlighting
                ext = Path(tool.arguments).suffix
                lang_map = {'.py': 'python', '.js': 'javascript', '.yaml': 'yaml', 
                           '.json': 'json', '.sh': 'bash', '.md': 'markdown'}
                lang = lang_map.get(ext, 'text')
                
                content = content[:2000]  # Truncate
                return f"```{lang}\n# {tool.arguments}\n{content}\n```"
            else:
                return f"❌ Could not read: {tool.arguments}"
        
        elif tool.tool_type == ToolType.LIST:
            parts = tool.arguments.split(maxsplit=1)
            directory = parts[0] if parts else "."
            pattern = parts[1] if len(parts) > 1 else "*"
            
            files = self.code_context.list_files(directory, pattern)
            if files:
                file_list = "\n".join(f"  • {f}" for f in files[:30])
                if len(files) > 30:
                    file_list += f"\n  ... and {len(files) - 30} more"
                return f"**Files in {directory}/{pattern}:**\n{file_list}"
            else:
                return f"No files found: {directory}/{pattern}"
        
        return ""
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """Process user input and generate response with streaming"""
        
        # Add to history
        self.messages.append({"role": "user", "content": user_input})
        
        # Build prompt
        prompt = self._format_messages_for_model(user_input)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with streaming
        response_parts = []
        
        if stream and RICH_AVAILABLE:
            # Rich streaming display
            with Live("", console=self.console, refresh_per_second=10) as live:
                for chunk in self.streamer.generate_stream(inputs, max_tokens=512):
                    response_parts.append(chunk)
                    current_text = "".join(response_parts)
                    
                    # Render markdown in real-time
                    try:
                        md = Markdown(current_text)
                        live.update(md)
                    except:
                        live.update(current_text)
        else:
            # Non-streaming fallback
            for chunk in self.streamer.generate_stream(inputs, max_tokens=512):
                response_parts.append(chunk)
                if not stream:
                    print(chunk, end="", flush=True)
        
        response = "".join(response_parts).strip()
        
        # Process tool calls
        tool_calls = self._extract_tool_calls(response)
        
        if tool_calls:
            if RICH_AVAILABLE:
                self.console.print("\n[dim]Executing tools...[/dim]")
            else:
                print("\n🔧 Executing tools...")
            
            # Execute tools and inject results
            for tool in tool_calls:
                if RICH_AVAILABLE:
                    self.console.print(f"[cyan]→ {tool.tool_type.value}: {tool.arguments}[/cyan]")
                else:
                    print(f"→ {tool.tool_type.value}: {tool.arguments}")
                
                tool_result = self._execute_tool(tool)
                
                # Replace tool call with result in response
                pattern = re.escape(f"[{tool.tool_type.value.upper()}: {tool.arguments}]")
                response = re.sub(pattern, f"\n{tool_result}\n", response, flags=re.IGNORECASE)
        
        # Add to history
        self.messages.append({"role": "assistant", "content": response})
        
        # Trim history if too long
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
        
        return response
    
    def run_interactive(self):
        """Run interactive chat loop"""
        
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "[bold cyan]SRE AI Assistant[/bold cyan]\n"
                "Commands: /help, /clear, /quit, /exec <cmd>, /read <file>",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print("SRE AI Assistant")
            print("Commands: /help, /clear, /quit, /exec <cmd>, /read <file>")
            print("=" * 60)
        
        while True:
            try:
                # Get input
                if RICH_AVAILABLE:
                    user_input = self.console.input("\n[bold green]You:[/bold green] ").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit', '/q']:
                        break
                    
                    elif user_input == '/clear':
                        self.messages = []
                        if RICH_AVAILABLE:
                            self.console.clear()
                        print("🗑️  Cleared history")
                        continue
                    
                    elif user_input == '/help':
                        help_text = """
**Commands:**
• `/quit` - Exit
• `/clear` - Clear history  
• `/exec <command>` - Run command directly
• `/read <file>` - Read file directly

**Usage:**
Just chat naturally! The assistant will use tools when needed.
Examples:
• "Show me all pods in the default namespace"
• "Read the config.yaml file"
• "What's the CPU usage?"
"""
                        if RICH_AVAILABLE:
                            self.console.print(Markdown(help_text))
                        else:
                            print(help_text)
                        continue
                    
                    elif user_input.startswith('/exec '):
                        cmd = user_input[6:]
                        result = self.executor.execute(cmd)
                        if result['success']:
                            print(f"```\n{result['stdout']}\n```")
                        else:
                            print(f"Error: {result['stderr']}")
                        continue
                    
                    elif user_input.startswith('/read '):
                        filepath = user_input[6:]
                        content = self.code_context.read_file(filepath)
                        if content:
                            print(f"```\n{content}\n```")
                        else:
                            print(f"❌ Could not read {filepath}")
                        continue
                
                # Generate response
                if RICH_AVAILABLE:
                    self.console.print("\n[bold cyan]Assistant:[/bold cyan]", end=" ")
                else:
                    print("\nAssistant: ", end="")
                
                response = self.chat(user_input, stream=True)
                
                # Display final response if not already shown via streaming
                if not RICH_AVAILABLE or not response:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Entry point"""
    
    # Find models
    models_dir = Path("./models")
    pkl_files = []
    
    if models_dir.exists():
        pkl_files = list(models_dir.glob("*/model_info.pkl"))
        root_pkl = models_dir / "model_info.pkl"
        if root_pkl.exists():
            pkl_files.append(root_pkl)
    
    if not pkl_files:
        print("❌ No cached models found")
        print("💡 Run your training script first")
        return
    
    # Select model
    if len(pkl_files) == 1:
        model_path = pkl_files[0]
    else:
        print("\n📦 Available models:\n")
        for i, p in enumerate(pkl_files, 1):
            print(f"  {i}. {p.parent.name}")
        
        choice = input(f"\nSelect (1-{len(pkl_files)}): ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(pkl_files):
            print("❌ Invalid choice")
            return
        
        model_path = pkl_files[int(choice) - 1]
    
    # Run
    try:
        interactor = SmartModelInteractor(str(model_path))
        interactor.run_interactive()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
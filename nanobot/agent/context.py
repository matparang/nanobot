"""Context builder for assembling agent prompts."""

import base64
import logging
import mimetypes
import platform
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import SuperpositionalState
from nanobot.agent.skills import SkillsLoader
from nanobot.runtime.state import state

logger = logging.getLogger(__name__)

# Bootstrap file base names (without extension)
BOOTSTRAP_NAMES = ["AGENTS", "SOUL", "USER", "TOOLS", "IDENTITY"]


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    
    Supports Triune Memory: prefers .yaml files for token efficiency,
    falls back to .md files for backward compatibility.
    """

    # Legacy constant for backward compatibility
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    def __init__(self, workspace: Path, memory_config: dict[str, Any] | None = None):
        self.workspace = workspace
        self.memory = MemoryStore(workspace, config=memory_config or {})
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        user_query: str = "",
        latent_state: SuperpositionalState | None = None,
    ) -> str:
        """
        Build the system prompt using the 6-Block Context Workflow.
        
        This implements the context-engineered workflow with:
        1. System & Persona (Static)
        2. Active Learning State & Resources (Dynamic)
        3. Tools/Skills (Capabilities)
        4. (History and User Message handled separately in build_messages)
        
        Args:
            skill_names: Optional list of skills to include.
            user_query: Current user query for relevant node retrieval.
        
        Returns:
            Complete system prompt with structured blocks.
        """
        parts = []

        # Block 1: System & Persona (Static)
        parts.append(self._get_identity())

        # Bootstrap files
        if state.triune_memory_enabled:
            bootstrap = self._load_bootstrap_files()
            if bootstrap:
                parts.append(bootstrap)

        if state.latent_reasoning_enabled and latent_state:
            top_intent = (
                latent_state.hypotheses[0].intent
                if latent_state.hypotheses
                else "proceed with standard processing"
            )
            parts.append(
                "<latent_state>\n"
                f"entropy: {latent_state.entropy}\n"
                f"strategy: {latent_state.strategic_direction}\n"
                f"primary_hypothesis: {top_intent}\n"
                "</latent_state>"
            )

        # Block 1.5: Cognitive Directive
        cognitive_directive = """# COGNITIVE DIRECTIVE

Memory retrieved in the RESOURCES & MEMORY section is authoritative internal knowledge.

You must use retrieved memory as primary reasoning substrate.

When answering questions:
- First consult retrieved memory
- Prefer memory over tools
- Prefer memory over assumptions
- Use tools only if memory does not contain the answer

If memory contains relevant explanation, use it directly.
Do not ignore relevant memory.
"""
        parts.append(cognitive_directive)

        # Block 2: Active Learning State & Resources (Dynamic)
        # Combines ALS + Top-K Fractal Nodes + Core Memory.md
        resource_parts = []

        # ALS Context
        als_content = self.memory.get_als_context()
        if als_content:
            resource_parts.append(als_content)

        # Core Memory (legacy MEMORY.md)
        core_memory = self.memory.read_long_term()
        if core_memory:
            resource_parts.append(f"## Long-term Memory\n{core_memory}")

        # Fractal Nodes (token-efficient top-K retrieval)
        if user_query and (state.mem0_enabled or state.fractal_memory_enabled):
            fractal_content = self.memory.retrieve_relevant_nodes(user_query, k=5)
            if fractal_content:
                resource_parts.append(fractal_content)
        if state.entangled_memory_enabled and user_query:
            entangled_nodes = self.memory.get_entangled_context(user_query, top_k=5)
            entangled_content = self.memory.format_nodes_for_context(entangled_nodes)
            if entangled_content:
                resource_parts.append(entangled_content)

        if resource_parts:
            parts.append("# RESOURCES & MEMORY\n\n" + "\n\n".join(resource_parts))

        # Block 3: Tools/Skills (Capabilities)
        # Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        import time as _time
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, think step by step: what you know, what you need, and why you chose this tool.
When remembering something important, write to {workspace_path}/memory/MEMORY.md
To recall past events, grep {workspace_path}/memory/HISTORY.md"""

    def _load_bootstrap_files(self) -> str:
        """
        Load all bootstrap files from workspace.
        
        Triune Memory: Prefers .yaml files for token efficiency,
        falls back to .md files for backward compatibility.
        """
        parts = []

        for base_name in BOOTSTRAP_NAMES:
            yaml_path = self.workspace / f"{base_name}.yaml"
            md_path = self.workspace / f"{base_name}.md"

            # Prefer YAML if available (token-efficient)
            if yaml_path.exists():
                try:
                    import yaml
                    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                    content = self._yaml_to_context(data)
                    parts.append(f"## {base_name}\n\n{content}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {yaml_path}: {e}")

            # Fall back to MD (backward compatibility)
            if md_path.exists():
                content = md_path.read_text(encoding="utf-8")
                parts.append(f"## {base_name}.md\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def _yaml_to_context(self, data: dict[str, Any]) -> str:
        """
        Convert YAML data to context string for LLM.
        
        Uses shared formatting from translator module.
        
        Args:
            data: YAML data dictionary
            
        Returns:
            Formatted context string
        """
        from nanobot.utils.translator import yaml_data_to_context

        if not isinstance(data, dict):
            return str(data)

        return yaml_data_to_context(data)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        latent_state: SuperpositionalState | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call using 6-block structure.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt with 6-block context.
        """
        messages = []

        # Block 1-3: System prompt (with user query for fractal node retrieval)
        system_prompt = self.build_system_prompt(
            skill_names, user_query=current_message, latent_state=latent_state
        )
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
        messages.append({"role": "system", "content": system_prompt})

        # Block 4: Assistant Messages / History (Short-term)
        messages.extend(history)

        # Block 5: User Message (The Trigger)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Thinking models reject history without this
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages

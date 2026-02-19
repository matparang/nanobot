"""Agent loop: the core processing engine."""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.dual_reasoning import DualReasoningOrchestrator
from nanobot.agent.latent import LatentReasoner
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config import settings
from nanobot.llm_adapter import LLMAdapter, LLMAccessDeniedError
from nanobot.middleware.rate_limiter import RateLimitConfig, RateLimiter
from nanobot.providers.base import LLMProvider
from nanobot.runtime.chi_tracker import ChiTracker
from nanobot.runtime.state import state
from nanobot.session.manager import Session, SessionManager
from nanobot.telemetry.exporter import MetricsExporter
from nanobot.telemetry.metrics import (
    active_sessions,
    latent_reasoning_duration,
    memory_ops_count,
    memory_retrieval_duration,
    tool_execution_count,
)
from nanobot.memory.consolidation import ConsolidationPipeline
from nanobot.memory.session_store import SessionStore


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        memory_config: dict[str, Any] | None = None,
        rate_limit_config: dict[str, Any] | None = None,
        telemetry_config: dict[str, Any] | None = None,
        enable_latent_reasoning: bool = True,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        # Wrap provider with LLM adapter for enforcement
        self.llm_adapter = LLMAdapter(provider)
        self.provider = provider  # Keep reference for backward compatibility
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.memory_config = memory_config or {}
        self.clarify_entropy_threshold = float(
            self.memory_config.get("clarify_entropy_threshold", settings.clarify_entropy_threshold)
        )
        self.max_context_nodes = int(
            self.memory_config.get("max_context_nodes", settings.max_context_nodes)
        )
        latent_timeout_seconds = int(
            self.memory_config.get("latent_timeout_seconds", settings.latent_timeout_seconds)
        )

        self.context = ContextBuilder(workspace, memory_config=memory_config)
        self.latent_engine = LatentReasoner(
            provider=self.provider,
            model=self.model,
            timeout_seconds=latent_timeout_seconds,
            memory_config=self.memory_config,
        )
        self.chi_tracker = ChiTracker()
        self.dual_reasoner = DualReasoningOrchestrator(
            provider=self.provider,
            model=self.model,
            chi_tracker=self.chi_tracker,
            memory_config=self.memory_config,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.episodic_store = SessionStore(self.workspace)
        self.consolidation_pipeline = ConsolidationPipeline(
            self.workspace,
            memory_config=self.memory_config,
        )

        self._episodic_enabled = bool(self.memory_config.get("episodic_enabled", True))
        self._auto_consolidate_enabled = bool(self.memory_config.get("auto_consolidate_enabled", True))
        self._auto_consolidate_event_threshold = int(self.memory_config.get("auto_consolidate_event_threshold", 40))

        # Wire memory-aware reasoning if episodic memory is enabled
        if self._episodic_enabled:
            try:
                from nanobot.runtime.state import state
                
                # Check if LLM is disabled - use v2 deterministic reasoner
                if not state.llm_enabled:
                    # Initialize v2 components for LLM-free operation
                    from nanobot.memory.relational_cache_v2 import RelationalCacheV2
                    from nanobot.memory.memory_first_reasoner_v2 import MemoryFirstReasonerV2
                    
                    # Create cache and reasoner instances
                    relational_cache_v2 = RelationalCacheV2()
                    self.reasoner = MemoryFirstReasonerV2(cache=relational_cache_v2)
                    
                    # Store for ingestion pipeline
                    self.relational_cache_v2 = relational_cache_v2
                    
                    logger.info("[Nanobot] Using deterministic memory-first reasoning (v2) - LLM disabled")
                else:
                    # LLM enabled - use v1 with HypothesisEngine
                    from nanobot.memory.memory_aware_reasoner import wrap_latent_reasoner_with_memory
                    
                    self.latent_engine = wrap_latent_reasoner_with_memory(
                        self.latent_engine,
                        workspace=self.workspace,
                        memory_config=self.memory_config,
                    )
                    
                    # Also wrap the dual reasoner's latent reasoner
                    self.dual_reasoner.latent_reasoner = wrap_latent_reasoner_with_memory(
                        self.dual_reasoner.latent_reasoner,
                        workspace=self.workspace,
                        memory_config=self.memory_config,
                    )
                    
                    logger.info("Memory-aware reasoning enabled (LatentReasoner wrapped with HypothesisEngine)")
            except Exception as e:
                logger.warning(f"Failed to enable memory-aware reasoning (non-fatal): {e}")

        self._running = False
        # consolidation_queue_size bounds background memory-consolidation backlog.
        self._consolidation_queue: asyncio.Queue[str | Session] = asyncio.Queue(
            maxsize=int(self.memory_config.get("consolidation_queue_size", 128))
        )
        self._consolidation_task: asyncio.Task | None = None
        rate_limit_config = rate_limit_config or {}
        self.rate_limiter = RateLimiter(
            RateLimitConfig(
                max_calls=int(rate_limit_config.get("max_calls", 10)),
                window_seconds=int(rate_limit_config.get("window_seconds", 60)),
                enabled=bool(rate_limit_config.get("enabled", True)),
            )
        )
        telemetry_config = telemetry_config or {}
        self.metrics_exporter = MetricsExporter(
            port=int(telemetry_config.get("port", 9090)),
            enabled=bool(telemetry_config.get("enabled", True)),
        )
        self._telemetry_started = False
        self._register_default_tools()

        # Initialize MCP integration
        self.mcp_registry = None
        self._mcp_configs = None
        if self.workspace:
            # Look for MCP config in workspace
            mcp_config_path = self.workspace / "nanobot" / "config" / "mcp.yaml"
            if mcp_config_path.exists():
                try:
                    from nanobot.mcp.config_loader import MCPConfigLoader
                    from nanobot.mcp.registry import MCPRegistry

                    self._mcp_configs = MCPConfigLoader.load(mcp_config_path)
                    if self._mcp_configs:
                        self.mcp_registry = MCPRegistry(self.tools)
                        logger.info(f"MCP integration initialized with {len(self._mcp_configs)} servers")
                except Exception as e:
                    logger.warning(f"MCP initialization failed: {e}")

    @property
    def enable_latent_reasoning(self) -> bool:
        """Get effective latent reasoning state from global runtime."""
        return state.latent_reasoning_enabled

    @property
    def has_v2_reasoner(self) -> bool:
        """Check if v2 reasoner is active."""
        return hasattr(self, 'reasoner') and self.reasoner is not None

    def _extract_relations_to_v2_cache(self, content: str) -> int:
        """Extract relations from text and add to v2 cache.
        
        Args:
            content: Text to extract relations from
            
        Returns:
            Number of relations extracted and added to cache
        """
        if not hasattr(self, 'relational_cache_v2') or not self.relational_cache_v2:
            return 0
            
        try:
            from nanobot.memory.relation_extractor_v2 import RelationExtractionEngineV2
            from nanobot.memory.types_v2 import IngestionStatus
            
            extractor_v2 = RelationExtractionEngineV2()
            lines = content.split('\n')
            extracted_count = 0
            for line in lines:
                result = extractor_v2.extract(line.strip())
                if result.status == IngestionStatus.ACCEPTED and result.relation:
                    a, relation_type, b = result.relation
                    self.relational_cache_v2.add_relation(
                        a, relation_type, b,
                        confidence=result.confidence, source="user_input"
                    )
                    extracted_count += 1
            return extracted_count
        except Exception as e:
            logger.warning(f"V2 relation extraction failed (non-fatal): {e}")
            return 0

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))

        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        if not self._telemetry_started:
            try:
                self.metrics_exporter.start()
                self._telemetry_started = True
            except OSError as exc:
                logger.warning(f"Metrics exporter failed to start: {exc}")
        if self._consolidation_task is None or self._consolidation_task.done():
            self._consolidation_task = asyncio.create_task(self._consolidation_worker())

        # Connect MCP clients
        if self.mcp_registry and self._mcp_configs:
            for config in self._mcp_configs:
                try:
                    await self.mcp_registry.add_client(config)
                except Exception as e:
                    logger.warning(f"Failed to connect MCP server '{config.name}': {e}")

        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

        # Cleanup: Disconnect MCP clients
        if self.mcp_registry:
            try:
                await self.mcp_registry.disconnect_all()
            except Exception as e:
                logger.warning(f"Error disconnecting MCP clients: {e}")

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
        logger.info("Agent loop stopping")

    def _try_v2_reasoning(self, query: str) -> str | None:
        """
        Try to answer a query using v2 memory-first reasoner.
        
        Returns:
            Answer string if v2 can answer, None otherwise
        """
        if not self.has_v2_reasoner:
            return None
            
        try:
            from nanobot.memory.types_v2 import TruthValue, RelationType
            import re
            
            # Parse comparative queries: "Who is taller, X or Y?"
            # Pattern 1: "Who is taller, X or Y?"
            match = re.search(r'who\s+is\s+(taller|shorter)\s*,?\s+(\w+)\s+or\s+(\w+)', query, re.IGNORECASE)
            if match:
                relation_word = match.group(1).lower()
                entity_a = match.group(2)
                entity_b = match.group(3)
                
                relation_type = RelationType.TALLER_THAN if relation_word == "taller" else RelationType.SHORTER_THAN
                
                # Query both directions
                result_ab = self.reasoner.query_pairwise(entity_a, entity_b, relation_type)
                result_ba = self.reasoner.query_pairwise(entity_b, entity_a, relation_type)
                
                if result_ab.value == TruthValue.TRUE:
                    return f"{entity_a} is {relation_word} than {entity_b}"
                elif result_ba.value == TruthValue.TRUE:
                    return f"{entity_b} is {relation_word} than {entity_a}"
                elif result_ab.value == TruthValue.FALSE:
                    # If A is not taller than B, then B is taller than A
                    opposite_word = "shorter" if relation_word == "taller" else "taller"
                    return f"{entity_b} is {opposite_word} than {entity_a}"
                else:
                    return None  # UNKNOWN
            
            # Pattern 2: "Is X taller than Y?"
            match = re.search(r'is\s+(\w+)\s+(taller|shorter)\s+than\s+(\w+)', query, re.IGNORECASE)
            if match:
                entity_a = match.group(1)
                relation_word = match.group(2).lower()
                entity_b = match.group(3)
                
                relation_type = RelationType.TALLER_THAN if relation_word == "taller" else RelationType.SHORTER_THAN
                result = self.reasoner.query_pairwise(entity_a, entity_b, relation_type)
                
                if result.value == TruthValue.TRUE:
                    return f"Yes, {entity_a} is {relation_word} than {entity_b}"
                elif result.value == TruthValue.FALSE:
                    return f"No, {entity_a} is not {relation_word} than {entity_b}"
                else:
                    return None  # UNKNOWN
            
            # Pattern 3: Superlative queries "Who is tallest/shortest?"
            match = re.search(r'who\s+is\s+(tallest|shortest)', query, re.IGNORECASE)
            if match:
                kind = match.group(1).lower()
                result = self.reasoner.query_superlative(kind)
                
                if result.value == TruthValue.TRUE:
                    return result.message
                else:
                    return None  # UNKNOWN
            
            # No pattern matched
            return None
            
        except Exception as e:
            logger.warning(f"V2 reasoning error: {e}")
            return None

    def _get_or_create_episodic_session_id(self, session) -> str:
        """
        Deterministic per-session-key episodic session id, stored in Session.metadata.
        """
        def _build_session_id() -> str:
            digest = hashlib.sha256(session.key.encode("utf-8")).hexdigest()[:12]
            created_ts = session.created_at.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"session_{created_ts}_{digest}"

        try:
            existing = (session.metadata or {}).get("episodic_session_id")
            if existing:
                return existing

            session_id = _build_session_id()

            session.metadata = session.metadata or {}
            session.metadata["episodic_session_id"] = session_id
            self.sessions.save(session)

            self.episodic_store.start(
                session_id=session_id,
                metadata={
                    "session_key": session.key,
                    "channel": getattr(session, "channel", None),
                },
            )
            return session_id
        except Exception as e:
            logger.warning(f"Episodic session id creation failed (non-fatal): {e}")
            return _build_session_id()

    def _append_episodic_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
        if not self._episodic_enabled:
            return
        try:
            self.episodic_store.append_event(session_id=session_id, event_type=event_type, payload=payload)
        except Exception as e:
            logger.warning(f"Episodic append_event failed (non-fatal): {e}")

    def _maybe_enqueue_consolidation(self, session_id: str) -> None:
        if not self._auto_consolidate_enabled:
            return
        try:
            session_data = self.episodic_store.load(session_id)
            event_count = len(session_data.get("events", []))
            if event_count < self._auto_consolidate_event_threshold:
                return
            try:
                self._consolidation_queue.put_nowait(session_id)
            except asyncio.QueueFull:
                logger.warning("Consolidation queue full; skipping auto-enqueue (non-fatal)")
        except Exception as e:
            logger.warning(f"Auto consolidation enqueue failed (non-fatal): {e}")

    def _build_outbound_with_event(
        self,
        msg: InboundMessage,
        content: str,
        episodic_session_id: str,
        tools_used: list[str] | None = None,
    ) -> OutboundMessage:
        outbound_metadata = dict(msg.metadata or {})
        if tools_used:
            outbound_metadata["tools_used"] = tools_used
        outbound = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata=outbound_metadata,  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
        self._append_episodic_event(
            session_id=episodic_session_id,
            event_type="interaction",
            payload={
                "user_message": msg.content,
                "agent_response": outbound.content or "",
                "tools_used": tools_used or [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        # Eager relation extraction from user message
        if self._episodic_enabled:
            try:
                from nanobot.memory.relation_extractor import RelationExtractionEngine
                
                extractor = RelationExtractionEngine()
                extracted_count = extractor.extract_and_ingest(
                    text=msg.content,
                    cache=self.consolidation_pipeline.relational_cache,
                    session_store=self.episodic_store,
                    session_id=episodic_session_id,
                )
                
                # Also feed v2 cache if it exists
                extracted_v2_count = self._extract_relations_to_v2_cache(msg.content)
                if extracted_v2_count > 0:
                    logger.debug(f"Fed {extracted_v2_count} relations to v2 cache")
                
                if extracted_count > 0:
                    logger.info(f"Eager relation extraction: {extracted_count} relations from user message")
            except Exception as e:
                logger.warning(f"Eager relation extraction failed (non-fatal): {e}")
        
        return outbound

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # [CLI] received prompt - log the incoming message
        logger.info(f"[CLI] Received prompt from {msg.channel}:{msg.sender_id}")
        content_preview = str(msg.content)[:100] if msg.content else ""
        logger.debug(f"[CLI] Message content: {content_preview}...")
        
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"[Nanobot] Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        rate_limit_key = f"{msg.channel}:{msg.chat_id}"
        if not await self.rate_limiter.is_allowed(rate_limit_key):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="⚠️ Rate limit exceeded. Please wait.",
                metadata=msg.metadata or {},
            )

        # Get or create session
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        episodic_session_id = self._get_or_create_episodic_session_id(session)

        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if self._auto_consolidate_enabled:
                    self.consolidation_pipeline.run_full_pipeline(
                        session_ids=[episodic_session_id],
                        archive_sessions=True,
                    )
            except Exception as e:
                logger.warning(f"Episodic consolidation on /new failed (non-fatal): {e}")

            await self._consolidate_memory(session, archive_all=True)
            session.clear()
            self.sessions.save(session)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 New session started. Memory consolidated.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        # Consolidate memory before processing if session is too large
        if len(session.messages) > self.memory_window:
            if self._running:
                try:
                    self._consolidation_queue.put_nowait(session)
                except asyncio.QueueFull:
                    logger.warning("Memory consolidation queue is full; skipping enqueue")
            else:
                await self._consolidate_memory(session)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(msg.channel, msg.chat_id)

        retrieval_start = time.time()
        latent_nodes = self.context.memory.get_entangled_context(msg.content, top_k=self.max_context_nodes)
        memory_retrieval_duration.observe(time.time() - retrieval_start)
        latent_context = self.context.memory._format_nodes(latent_nodes)
        latent_state = None
        should_clarify = False
        if state.dual_layer_enabled:
            latent_start = time.time()
            dual_result = await self.dual_reasoner.reason(
                user_message=msg.content,
                context_summary=latent_context,
            )
            latent_reasoning_duration.observe(time.time() - latent_start)
            latent_state = dual_result.final_state
            should_clarify = (
                dual_result.system_used != "system1"
                and latent_state.entropy > self.clarify_entropy_threshold
            )
            if state.reasoning_audit_enabled:
                from nanobot.cli.audit import AuditAction, audit_log

                audit_log(
                    AuditAction.REASONING_COMPLETED,
                    {
                        "system_used": dual_result.system_used,
                        "confidence": (
                            dual_result.system1_result.confidence
                            if dual_result.system1_result
                            else 0.0
                        ),
                        "entropy": latent_state.entropy,
                        "chi_cost": dual_result.total_chi_cost,
                        "latency_ms": dual_result.latency_ms,
                        "escalated": dual_result.escalated,
                        "hypothesis_count": len(latent_state.hypotheses),
                        "pattern_cache_hit": (
                            dual_result.system1_result.pattern_hit
                            if dual_result.system1_result
                            else False
                        ),
                        "correctness": None,
                    },
                    source="dual_reasoning",
                )
                if dual_result.escalated:
                    audit_log(
                        AuditAction.REASONING_ESCALATED,
                        {"trace": dual_result.reasoning_trace},
                        source="dual_reasoning",
                    )
                if state.chi_tracking_enabled:
                    audit_log(
                        AuditAction.CHI_BUDGET_UPDATE,
                        self.chi_tracker.get_budget_status(),
                        source="dual_reasoning",
                    )
        elif self.enable_latent_reasoning:
            latent_start = time.time()
            latent_state = await self.latent_engine.reason(user_message=msg.content, context_summary=latent_context)
            latent_reasoning_duration.observe(time.time() - latent_start)
            should_clarify = latent_state.entropy > self.clarify_entropy_threshold
        if should_clarify and latent_state:
            if len(latent_state.hypotheses) >= 2:
                opt1 = latent_state.hypotheses[0].intent
                opt2 = latent_state.hypotheses[1].intent
            else:
                opt1 = "continue with your current request"
                opt2 = "provide a bit more detail"
            outbound = self._build_outbound_with_event(
                msg,
                (
                    "I'm detecting some ambiguity. "
                    f"Are you looking to {opt1}, or {opt2}? Could you clarify?"
                ),
                episodic_session_id,
                tools_used=[],
            )
            self._maybe_enqueue_consolidation(episodic_session_id)
            return outbound

        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            latent_state=latent_state,
        )

        # Agent loop
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        active_sessions.set(len(self.sessions._cache))

        while iteration < self.max_iterations:
            iteration += 1

            # Check if v2 reasoner can answer the query (when LLM is disabled)
            if self.has_v2_reasoner and not state.llm_enabled:
                try:
                    v2_answer = self._try_v2_reasoning(msg.content)
                    if v2_answer:
                        logger.info(f"[Nanobot] V2 reasoner provided answer: {v2_answer}")
                        final_content = v2_answer
                        break
                except Exception as e:
                    logger.warning(f"V2 reasoning attempt failed (non-fatal): {e}")

            # [Nanobot] routing decision - log before calling LLM
            logger.info(f"[Nanobot] Routing decision: calling LLM adapter (iteration {iteration}/{self.max_iterations})")
            logger.debug(f"[Nanobot] LLM enabled: {state.llm_enabled}, Model: {self.model}")
            
            try:
                # Call LLM through adapter (enforces global LLM_ENABLED flag)
                response = await self.llm_adapter.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature
                )
            except LLMAccessDeniedError as e:
                # LLM is disabled - but still extract relations if v2 is active
                logger.warning(f"[Nanobot] LLM access denied: {str(e)}")
                
                # Extract relations from user message using helper method
                extracted_v2_count = self._extract_relations_to_v2_cache(msg.content)
                if extracted_v2_count > 0:
                    logger.info(f"Extracted {extracted_v2_count} relations to v2 cache (LLM disabled)")
                
                # Try v2 reasoning one more time after extraction
                v2_answer = None
                if self.has_v2_reasoner:
                    try:
                        v2_answer = self._try_v2_reasoning(msg.content)
                    except Exception as ex:
                        logger.warning(f"V2 reasoning after extraction failed: {ex}")
                
                if v2_answer:
                    final_content = v2_answer
                    # Save to session
                    session.add_message("user", msg.content)
                    session.add_message("assistant", final_content)
                    self.sessions.save(session)
                    
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=final_content,
                        metadata=msg.metadata or {},
                    )
                
                # No v2 answer available - return generic message
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        "🤖 Nanobot (Memory-Only Mode)\n\n"
                        "LLM access is currently disabled. Operating in deterministic memory-only mode.\n"
                        f"To enable LLM, use the --enable-llm flag or enable it through configuration."
                    ),
                    metadata=msg.metadata or {},
                )

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    try:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_execution_count.labels(tool_name=tool_call.name, status="success").inc()
                    except Exception:
                        tool_execution_count.labels(tool_name=tool_call.name, status="error").inc()
                        raise
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                # Interleaved CoT: reflect before next action
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                # No tool calls, we're done
                final_content = response.content
                break

        if final_content is None:
            if iteration >= self.max_iterations:
                final_content = f"Reached {self.max_iterations} iterations without completion."
            else:
                final_content = "I've completed processing but have no response to give."

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")

        # Save to session (include tool names so consolidation sees what happened)
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # --- NEW: Fractal Memory Reflection Hook ---
        # Trigger reflection if significant tools were used or task completed
        await self._trigger_reflection(msg.content, final_content, tools_used)

        outbound = self._build_outbound_with_event(
            msg,
            final_content,
            episodic_session_id,
            tools_used=tools_used,
        )

        try:
            hypotheses_payload = []
            if latent_state and getattr(latent_state, "hypotheses", None):
                for h in getattr(latent_state, "hypotheses", []):
                    try:
                        if hasattr(h, "model_dump"):
                            hypotheses_payload.append(h.model_dump())
                        elif hasattr(h, "dict"):
                            hypotheses_payload.append(h.dict())
                        else:
                            hypotheses_payload.append(str(h))
                    except Exception as exc:
                        logger.warning(f"Hypothesis serialization failed (non-fatal): {exc}")
                        hypotheses_payload.append(str(h))
            metadata = outbound.metadata or {}
            entropy = getattr(latent_state, "entropy", None) if latent_state else metadata.get("entropy")
            strategic_direction = (
                getattr(latent_state, "strategic_direction", "")
                if latent_state
                else metadata.get("strategic_direction", "")
            )
            if not hypotheses_payload and "hypotheses" in metadata:
                hypotheses_payload = metadata.get("hypotheses", [])
            if hypotheses_payload or entropy is not None or strategic_direction:
                payload: dict[str, Any] = {
                    "hypotheses": hypotheses_payload,
                    "strategic_direction": strategic_direction,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if entropy is not None:
                    payload["entropy"] = entropy
                self._append_episodic_event(
                    session_id=episodic_session_id,
                    event_type="reasoning",
                    payload=payload,
                )
        except Exception as e:
            logger.warning(f"Reasoning episodic event capture failed (non-fatal): {e}")

        self._maybe_enqueue_consolidation(episodic_session_id)

        return outbound

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        if not await self.rate_limiter.is_allowed(f"{origin_channel}:{origin_chat_id}"):
            return OutboundMessage(
                channel=origin_channel,
                chat_id=origin_chat_id,
                content="⚠️ Rate limit exceeded. Please wait.",
            )

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        cron_tool = self.tools.get("cron")
        if isinstance(cron_tool, CronTool):
            cron_tool.set_context(origin_channel, origin_chat_id)

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            try:
                response = await self.llm_adapter.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature
                )
            except LLMAccessDeniedError:
                # LLM disabled - cannot process system message
                logger.warning("[Nanobot] Cannot process system message: LLM disabled")
                return None

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    try:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                        tool_execution_count.labels(tool_name=tool_call.name, status="success").inc()
                    except Exception:
                        tool_execution_count.labels(tool_name=tool_call.name, status="error").inc()
                        raise
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                # Interleaved CoT: reflect before next action
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )

    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md, then trim session."""
        if not session.messages:
            return
        memory = MemoryStore(self.workspace)
        if archive_all:
            old_messages = session.messages
            keep_count = 0
        else:
            keep_count = min(10, max(2, self.memory_window // 2))
            old_messages = session.messages[:-keep_count]
        if not old_messages:
            return
        logger.info(f"Memory consolidation started: {len(session.messages)} messages, archiving {len(old_messages)}, keeping {keep_count}")
        memory_ops_count.labels(operation="consolidate", status="started").inc()

        # Format messages for LLM (include tool names when available)
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            # Attempt LLM call for memory consolidation
            response = await self.llm_adapter.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
        except LLMAccessDeniedError:
            # LLM disabled - skip consolidation
            logger.warning("[Nanobot] Memory consolidation skipped: LLM disabled")
            return
        
        # Parse and apply the consolidation response
        try:
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            session.messages = session.messages[-keep_count:] if keep_count else []
            self.sessions.save(session)
            logger.info(f"Memory consolidation done, session trimmed to {len(session.messages)} messages")
            memory_ops_count.labels(operation="consolidate", status="success").inc()
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            memory_ops_count.labels(operation="consolidate", status="error").inc()

    async def _consolidation_worker(self) -> None:
        """Background worker for non-blocking memory consolidation."""
        while self._running:
            try:
                queue_item = await asyncio.wait_for(self._consolidation_queue.get(), timeout=1.0)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                continue
            try:
                if isinstance(queue_item, str):
                    self.consolidation_pipeline.run_full_pipeline(session_ids=[queue_item], archive_sessions=False)
                    continue
                await self._consolidate_memory(queue_item)
            except Exception as e:
                logger.warning(f"Background consolidation failed (non-fatal): {e}")

    async def _trigger_reflection(
        self,
        user_message: str,
        assistant_response: str,
        tools_used: list[str]
    ) -> None:
        """
        Trigger Fractal Memory reflection and ALS updates.
        
        This method decides whether to save a fractal node based on:
        - Keywords in the conversation ("remember", "important", "learn")
        - Significant tool usage (write_file, exec, etc.)
        - Task completion indicators
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            tools_used: List of tools that were executed
        """
        # Keywords that trigger memory capture
        memory_keywords = ["remember", "important", "learned", "note", "save this"]
        should_remember = any(kw in user_message.lower() for kw in memory_keywords)

        # Significant tools that warrant memory capture
        significant_tools = ["write_file", "exec", "spawn"]
        used_significant_tools = any(tool in tools_used for tool in significant_tools)

        # Only trigger reflection if there's something worth remembering
        if not (should_remember or used_significant_tools):
            return

        try:
            memory = MemoryStore(self.workspace)

            # Extract tags from user message (simple keyword extraction)
            words = user_message.lower().split()
            tags = [w.strip(".,!?;:") for w in words if len(w) > 4][:5]

            # Create a summary
            summary = user_message[:100] + ("..." if len(user_message) > 100 else "")

            # Determine content based on context
            if should_remember:
                content = f"User request: {user_message}\nAgent response: {assistant_response[:200]}"
            else:
                content = f"Task completed using {', '.join(tools_used)}: {summary}"

            # Save fractal node
            node = memory.save_fractal_node(
                content=content,
                tags=tags,
                summary=summary
            )

            # Update ALS with reflection
            memory.update_als(
                reflection=f"Completed interaction involving {', '.join(tools_used) if tools_used else 'conversation'}"
            )

            logger.info(f"Reflection captured: node {node.id}")

        except Exception as e:
            logger.warning(f"Reflection failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )

        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""

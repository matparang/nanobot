# Strict Nanobot LLM Enforcement - Implementation Summary

## Overview
This implementation adds strict enforcement to ensure that all user prompts are processed through Nanobot first, with LLM access being optional and controllable via runtime flags.

## Architecture

### 1. LLM Adapter Layer (`nanobot/llm_adapter.py`)
The `LLMAdapter` class serves as the single entry point for all LLM access:

```python
from nanobot.llm_adapter import LLMAdapter

# Wrap the provider
adapter = LLMAdapter(provider)

# All LLM calls go through the adapter
response = await adapter.chat(messages=messages)
```

**Key Features:**
- Enforces global `state.llm_enabled` flag before any LLM call
- Raises `LLMAccessDeniedError` when LLM is disabled
- Comprehensive logging for execution chain tracking
- Thread-safe state checking

### 2. Hard Block in Provider (`nanobot/providers/litellm_provider.py`)
A secondary enforcement layer in the provider itself:

```python
# In LiteLLMProvider.chat()
if not state.llm_enabled:
    logger.error("[LiteLLM Provider] HARD BLOCK: Direct LLM provider access denied")
    return LLMResponse(content="Error: LLM access denied", finish_reason="error")
```

**Purpose:**
- Catches any direct provider calls that bypass the adapter
- Returns error response instead of raising exception
- Provides clear error messages for debugging

### 3. AgentLoop Integration (`nanobot/agent/loop.py`)
The main processing loop uses the adapter:

```python
class AgentLoop:
    def __init__(self, bus, provider, workspace, ...):
        # Wrap provider with adapter
        self.llm_adapter = LLMAdapter(provider)
        self.provider = provider  # Keep for backward compatibility
        
    async def _process_message(self, msg):
        try:
            response = await self.llm_adapter.chat(messages=messages, ...)
        except LLMAccessDeniedError as e:
            # Return user-friendly error message
            return OutboundMessage(content="🤖 Nanobot (Memory-Only Mode)...")
```

## Execution Flow

### Normal Flow (LLM Enabled)
```
1. [CLI] Received prompt from cli:user
2. [Nanobot] Processing message from cli:user: <preview>
3. [Nanobot] Routing decision: calling LLM adapter (iteration 1/20)
4. [LLM Adapter] Calling LLM provider (model=<model>)
5. [LiteLLM Provider] Processing chat request (model=<model>)
6. [LLM Adapter] LLM call completed (finish_reason=stop)
7. Response returned to user
```

### LLM Disabled Flow
```
1. [CLI] Received prompt from cli:user
2. [Nanobot] Processing message from cli:user: <preview>
3. [Nanobot] Routing decision: calling LLM adapter (iteration 1/20)
4. [LLM Adapter] LLM access denied: LLM calls are globally disabled
5. [Nanobot] LLM access denied: <error message>
6. User receives: "🤖 Nanobot (Memory-Only Mode)
   LLM access is currently disabled. Operating in deterministic memory-only mode.
   To enable LLM, use the --enable-llm flag or enable it through configuration."
```

## Runtime Control

### CLI Flags
```bash
# Enable LLM
nanobot agent --enable-llm -m "Hello"

# Disable LLM (memory-only mode)
nanobot agent --disable-llm -m "Hello"

# Check status
nanobot state inspect
```

### Programmatic Control
```python
from nanobot.runtime.state import state

# Disable LLM
state.llm_enabled = False

# Enable LLM
state.llm_enabled = True

# Check status
if state.llm_enabled:
    print("LLM is enabled")
```

## Security Features

1. **Two-Layer Enforcement**
   - Primary: `LLMAdapter.chat()` raises exception
   - Secondary: `LiteLLMProvider.chat()` returns error

2. **Thread-Safe State**
   - `RuntimeState` uses locks for all state access
   - Singleton pattern ensures consistency

3. **Clear Error Messages**
   - Errors indicate exactly what went wrong
   - Provide actionable solutions (use --enable-llm)

4. **Audit Trail**
   - All LLM calls logged
   - Access denials logged with context

## Testing

### Unit Tests (12 tests)
- `tests/test_llm_enforcement.py`
  - LLM adapter enforcement
  - State toggling
  - Provider hard block
  - Error handling

### Integration Tests (3 tests)
- `tests/test_llm_integration.py`
  - End-to-end flow with LLM enabled
  - End-to-end flow with LLM disabled
  - Adapter usage verification

### Running Tests
```bash
# Run all LLM enforcement tests
pytest tests/test_llm_enforcement.py tests/test_llm_integration.py -v

# Run specific test
pytest tests/test_llm_enforcement.py::TestLLMAdapter::test_chat_when_llm_disabled -v

# Run with coverage
pytest tests/test_llm_enforcement.py --cov=nanobot.llm_adapter
```

## Design Decisions

### Why Two Enforcement Layers?
1. **Primary (LLMAdapter)**: Raises exception for proper error handling
2. **Secondary (Provider)**: Safety net for direct calls, returns error response

### Why Keep `self.provider` Reference?
- Backward compatibility with components that may access provider directly
- Allows gradual migration if needed
- Secondary enforcement layer catches any direct access

### Why Not Wrap Internal Components?
- Internal components (light_reasoner, dual_reasoner) use provider for heuristics
- Their calls are also blocked by the hard block in provider
- Minimal changes approach - only modified essential files

## Migration Guide

### Existing Code
If you have existing code that calls the provider directly:

```python
# OLD - Direct provider call (still works but logged as warning)
response = await provider.chat(messages=messages)

# NEW - Use adapter (recommended)
response = await llm_adapter.chat(messages=messages)
```

### Error Handling
```python
from nanobot.llm_adapter import LLMAccessDeniedError

try:
    response = await llm_adapter.chat(messages=messages)
except LLMAccessDeniedError:
    # Handle LLM disabled case
    return "LLM is currently disabled"
```

## Future Enhancements

1. **Per-User LLM Control**: Allow different users to have different LLM access
2. **Rate Limiting**: Limit LLM calls per time period
3. **Cost Tracking**: Track LLM usage costs
4. **Fallback Models**: Use cheaper models when primary is disabled
5. **Audit Dashboard**: Visualize LLM usage patterns

## Troubleshooting

### Q: LLM calls still happening when disabled?
A: Check internal components (light_reasoner, dual_reasoner) - they may be calling provider. The hard block will catch these.

### Q: How to check current LLM state?
A: Run `nanobot state inspect` or check `state.llm_enabled` programmatically.

### Q: Can I enable/disable LLM at runtime?
A: Yes! Use CLI flags or set `state.llm_enabled = True/False`.

### Q: What happens to memory consolidation when LLM disabled?
A: Memory consolidation is skipped when LLM is disabled (requires LLM for processing).

## Performance Impact

- **Negligible overhead**: Simple boolean check before each LLM call
- **Thread-safe**: Uses locks but minimal contention
- **Logging**: Only DEBUG level logging, can be disabled if needed

## Security Considerations

- All code changes scanned with CodeQL (0 alerts)
- No security vulnerabilities introduced
- Clear error messages don't leak sensitive information
- State changes are thread-safe and atomic

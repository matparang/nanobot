# Nanobot + Ollama Integration Plan (QWEN 2.5 7B Instruct)

This document outlines a stepwise plan to connect Nanobot with a local Ollama deployment running **QWEN 2.5 7B Instruct** (Q4 quantized). It focuses on keeping changes minimal, leveraging existing provider abstractions, and preserving memory across updates or reinstalls.

---

## 1. Current Architecture Snapshot

**LLM integration**
- Providers live in `nanobot/providers/` with a shared `LLMProvider` base.
- `litellm_provider.py` routes to OpenAI-compatible endpoints (vLLM, Ollama, etc.).
- `registry.py` auto-detects providers via config name, API base, or model keywords.

**Memory persistence**
- Sessions: `~/.nanobot/sessions/` (JSONL) via `nanobot/session/manager.py`.
- Long-term memory: `~/.nanobot/workspace/memory/` managed by `nanobot/agent/memory.py`
  - `MEMORY.md`, `HISTORY.md`, `lesson_X.json`, `fractal_index.json`, optional mem0 vectors.

---

## 2. Ollama Service Layout

Target host: Ubuntu 16GB RAM (no GPU required for Q4).

```
┌─────────────────────────────────────────────────┐
│ systemd: ollama.service                         │
│  ├─ API: http://localhost:11434                 │
│  └─ OpenAI-style endpoint: /v1/chat/completions │
├─────────────────────────────────────────────────┤
│ Models: qwen2.5:7b-instruct-q4_K_M (~4.5GB RAM) │
├─────────────────────────────────────────────────┤
│ Nanobot venv: ~/.nanobot/venv (nanobot + litellm│
│ + httpx + mem0 optional)                        │
└─────────────────────────────────────────────────┘
```

### Setup commands
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl enable --now ollama

# Pull model
ollama pull qwen2.5:7b-instruct-q4_K_M

# Smoke test native API
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","prompt":"Hello","stream":false}'
```

---

## 3. Provider Strategy

### Preferred: LiteLLM gateway (no new provider needed)
- Configure `providers.ollama` with `apiBase: "http://localhost:11434/v1"` and `apiKey: "local"`.
- Set default agent model to `qwen2.5:7b-instruct-q4_K_M`.
- Works with existing `LiteLLMProvider` and registry `is_local` detection.

Example `~/.nanobot/config.json`:
```json
{
  "providers": {
    "ollama": {
      "apiKey": "local",
      "apiBase": "http://localhost:11434/v1",
      "extraHeaders": {}
    }
  },
  "agents": {
    "defaults": {
      "provider": "ollama",
      "model": "qwen2.5:7b-instruct-q4_K_M",
      "memoryWindow": 50
    }
  }
}
```

### Fallback: Custom provider (only if LiteLLM path fails)
- Add `OllamaProvider` inheriting `LLMProvider`, posting to `/chat/completions`.
- Register in `registry.py` with `name="ollama"`, `is_local=True`, `default_api_base="http://localhost:11434/v1"`, `default_model="qwen2.5:7b-instruct"`.
- Keep timeout generous (300s) for local inference.

---

## 4. Venv and Dependencies

```bash
python3.11 -m venv ~/.nanobot/venv
source ~/.nanobot/venv/bin/activate
pip install --upgrade pip
pip install nanobot-ai litellm httpx  # mem0 optional for embeddings
```

Optional helper script:
```bash
#!/bin/bash
source ~/.nanobot/venv/bin/activate
export PYTHONPATH=\"$PYTHONPATH:$HOME/.nanobot/workspace\"
exec \"$@\"
```

---

## 5. Memory Persistence Plan

**Directories (persist between updates)**
- `~/.nanobot/sessions/` — per-channel JSONL logs.
- `~/.nanobot/workspace/memory/` — `MEMORY.md`, `HISTORY.md`, fractal lessons, `fractal_index.json`, `ALS.json`, optional mem0 store.
- Ollama cache: `/usr/share/ollama/models` (or `OLLAMA_HOME` override).

**Consolidation hook (pre-update)**
- Iterate active sessions, summarize last N messages into fractal memory, clear session files.
- Run importance decay/compaction on fractal index.

**Backup cron (daily)**
```bash
tar czf ~/backups/nanobot_memory_$(date +%Y%m%d_%H%M%S).tar.gz \
  -C ~/.nanobot workspace/memory sessions config.json
find ~/backups -name 'nanobot_memory_*.tar.gz' -mtime +30 -delete
```

---

## 6. Chess-Specific Context (QWEN 2.5 7B)

Goal: retain aggressive play style for TanyalahD persona.

**Memory usage**
- Tag fractal nodes with `["chess", "analysis"]` and optional `game:<id>`.
- Retrieve context with queries like `"aggressive tactical sacrifices winning attack"`.

**System prompt skeleton**
```
You are TanyalahD, an aggressive chess AI powered by QWEN 2.5 7B.
Best Move: <uci>
Reason: <1-2 sentences>
Risk Level: <LOW|MEDIUM|HIGH>
Confidence: <0-100>
Memory Context:
{AGGRESSIVE_PLAY_CONTEXT}
```

---

## 7. Validation Steps

1) **Service check**: `systemctl status ollama` and `curl http://localhost:11434/v1/models`.
2) **Nanobot smoke test**: `nanobot agent -m "What is 2+2?"` with Ollama config.
3) **Latency monitor**: watch `free -h` / `top -p $(pgrep ollama)` during inference.
4) **Memory persistence**: start session, restart Nanobot, confirm recall from `sessions/` or `MEMORY.md`.
5) **Chess demo (optional)**: load FEN and request best move; confirm context injection uses aggressive prompt.

---

## 8. Rollout Phases (minimal-impact)

- **Day 1:** Install Ollama, pull QWEN model, create venv, run smoke test.
- **Days 2-3:** Configure provider via LiteLLM; add custom provider only if needed.
- **Days 3-4:** Verify memory directories, run consolidation + backup scripts.
- **Days 5-6:** Add chess prompt/context wiring and test short game analyses.

Success criteria: local completions respond; Nanobot routes via Ollama; memory persists across restarts; chess reasoning remains aggressive with recalled context.

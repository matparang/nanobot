# Nanobot Configuration Guide

This guide explains how to configure Nanobot using the `config.json` file.

## Quick Start

1. **Copy the example config:**
   ```bash
   mkdir -p ~/.nanobot
   cp config.example.json ~/.nanobot/config.json
   ```

2. **Add your API keys:**
   Edit `~/.nanobot/config.json` and replace the placeholder API keys with your actual keys.

3. **Choose your LLM provider:**
   - For OpenAI: Set `providers.openai.apiKey` and use models like `gpt-4` or `gpt-3.5-turbo`
   - For Gemini: Set `providers.gemini.apiKey` and use models like `gemini/gemini-2.0-flash-exp`
   - For multiple providers: Set API keys for all providers you want to use

## Configuration File Location

Default location: `~/.nanobot/config.json`

The config file uses **camelCase** for JSON keys (e.g., `apiKey`, `maxTokens`) which are automatically converted to **snake_case** internally (e.g., `api_key`, `max_tokens`).

## Configuration Structure

### 1. Agents Configuration

Controls the default behavior of the AI agent:

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.nanobot/workspace",
      "model": "gemini/gemini-2.0-flash-exp",
      "maxTokens": 8192,
      "temperature": 0.7,
      "maxToolIterations": 20,
      "memoryWindow": 50
    }
  }
}
```

**Fields:**
- `workspace`: Directory where the agent can read/write files
- `model`: Default LLM model to use (format: `provider/model-name`)
- `maxTokens`: Maximum tokens in response
- `temperature`: Creativity level (0.0-1.0)
- `maxToolIterations`: Max number of tool calls per conversation turn
- `memoryWindow`: Number of recent messages to keep in context

### 2. Providers Configuration

Configure API keys and endpoints for different LLM providers:

```json
{
  "providers": {
    "openai": {
      "apiKey": "sk-YOUR_OPENAI_API_KEY_HERE",
      "apiBase": null,
      "extraHeaders": null
    },
    "gemini": {
      "apiKey": "YOUR_GEMINI_API_KEY_HERE",
      "apiBase": null,
      "extraHeaders": null
    }
  }
}
```

**Supported Providers:**
- `openai` - OpenAI API (GPT-4, GPT-3.5, etc.)
- `gemini` - Google Gemini API
- `anthropic` - Anthropic Claude API
- `openrouter` - OpenRouter (gateway for many models)
- `deepseek` - DeepSeek API
- `groq` - Groq API (fast inference)
- `zhipu` - Zhipu AI (GLM models)
- `dashscope` - Alibaba DashScope (Qwen models)
- `moonshot` - Moonshot AI (Kimi)
- `minimax` - MiniMax API
- `vllm` - Local vLLM deployment
- `aihubmix` - AiHubMix gateway
- `custom` - Any OpenAI-compatible endpoint

**Provider Fields:**
- `apiKey`: Your API key (required)
- `apiBase`: Custom API endpoint URL (optional, most providers have defaults)
- `extraHeaders`: Additional HTTP headers (optional, e.g., for custom auth)

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://aistudio.google.com/app/apikey
- Anthropic: https://console.anthropic.com/
- OpenRouter: https://openrouter.ai/keys (supports many models with one key)

### 3. Memory Configuration

Configure the Fractal Memory system and Active Learning State (ALS):

```json
{
  "memory": {
    "enabled": true,
    "provider": "local",
    "topK": 5,
    "archiveDir": "archives",
    "alsEnabled": true,
    "mem0ApiKey": "",
    "mem0UserId": "nanobot_user",
    "mem0OrgId": "",
    "mem0ProjectId": "",
    "mem0Version": "v1.1",
    "embeddingModel": "text-embedding-3-small",
    "embeddingDim": 1536,
    "useHybridSearch": true,
    "latentRetryAttempts": 3,
    "latentRetryMinWait": 1.0,
    "latentRetryMaxWait": 5.0,
    "latentRetryMultiplier": 1.0,
    "use_memory_v2": false,
    "confidence_threshold": 0.9
  }
}
```

**Core Fields:**
- `enabled`: Enable/disable memory system
- `provider`: Memory provider (`"local"` or `"mem0"`)
- `topK`: Number of relevant memories to retrieve
- `archiveDir`: Directory for lesson archives (relative to workspace)
- `alsEnabled`: Enable Active Learning State tracking

**V2 Memory-First Reasoning (Experimental):**
- `use_memory_v2`: Enable deterministic v2 memory-first reasoning (default: `false`)
- `confidence_threshold`: Minimum confidence for relation extraction in v2 mode (default: `0.9`)

When `use_memory_v2` is enabled:
- Queries are answered deterministically from the relational memory graph (no LLM fallback)
- Relation extraction uses confidence-gating (only accepts relations with confidence ≥ threshold)
- Returns `UNKNOWN` for ambiguous queries, self-comparisons, and incomplete information
- Supports comparative reasoning: pairwise queries, superlatives (tallest/shortest), transitive inference
- Automatically stores inverse relations and performs cycle detection
- Tiered ranking for partial orderings

When `use_memory_v2` is disabled (default):
- Uses v1 hypothesis engine with entropy-based LLM fallback
- Supports broader query types with probabilistic reasoning

**Local Memory Provider (Default):**
- Uses local files: `MEMORY.md`, `ALS.json`, `fractal_index.json`
- Stores lesson archives in `archives/lesson_*.json`
- No API key required
- Supports hybrid search (keyword + vector)

**mem0 Cloud Provider (Optional):**
Configure these fields if using mem0 (https://mem0.ai):
- `mem0ApiKey`: Your mem0 API key
- `mem0UserId`: User identifier for mem0
- `mem0OrgId`: Organization ID (optional)
- `mem0ProjectId`: Project ID (optional)
- `mem0Version`: API version

**Embedding Settings:**
- `embeddingModel`: OpenAI embedding model (requires OpenAI API key)
- `embeddingDim`: Embedding dimension (1536 for text-embedding-3-small)
- `useHybridSearch`: Combine keyword and vector search

**Latent Retry Settings:**
- `latentRetryAttempts`: Number of retry attempts for latent LLM calls
- `latentRetryMinWait`: Minimum backoff wait (seconds)
- `latentRetryMaxWait`: Maximum backoff wait (seconds)
- `latentRetryMultiplier`: Exponential backoff multiplier

**Note on Sentence Transformers:**
Sentence Transformers is a Python library for generating embeddings locally. To use it:
1. Install: `pip install sentence-transformers`
2. Set `embeddingModel` to a HuggingFace model like `"sentence-transformers/all-MiniLM-L6-v2"`
3. No API key needed - runs locally

However, the current implementation uses OpenAI embeddings by default. For local embeddings, you would need to modify the `MemoryStore` class to support sentence-transformers.

### 4. Channels Configuration

Enable chat integrations (WhatsApp, Telegram, Discord, etc.):

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allowFrom": ["user123", "user456"],
      "proxy": null
    }
  }
}
```

See individual channel documentation for setup instructions.

### 5. Tools Configuration

Configure available tools:

```json
{
  "tools": {
    "web": {
      "search": {
        "apiKey": "YOUR_BRAVE_SEARCH_API_KEY",
        "maxResults": 5
      }
    },
    "exec": {
      "timeout": 60
    },
    "restrictToWorkspace": false
  }
}
```

**Fields:**
- `web.search.apiKey`: Brave Search API key (get from https://brave.com/search/api/)
- `web.search.maxResults`: Max search results to return
- `exec.timeout`: Shell command timeout in seconds
- `restrictToWorkspace`: Restrict file access to workspace only

### 6. Memory Configuration

Configure memory and reasoning behavior:

```json
{
  "memory": {
    "enabled": true,
    "provider": "local",
    "useMemoryV2": false,
    "enableLlmFallback": true,
    "clarifyEntropyThreshold": 0.8
  }
}
```

**Fields:**
- `enabled`: Enable memory system (default: `true`)
- `provider`: Memory provider (`"local"` or `"mem0"`)
- `useMemoryV2`: Use v2 deterministic memory-first reasoning (default: `false`)
- `enableLlmFallback`: Allow LLM fallback when memory cannot answer (default: `true`)
- `clarifyEntropyThreshold`: Entropy threshold for LLM invocation (default: `0.8`)

**LLM Disable Mode:**

You can disable all LLM calls and use only memory-based deterministic reasoning:

1. **Via CLI flag** (overrides config):
   ```bash
   nanobot agent --disable-llm
   nanobot gateway --disable-llm
   ```

2. **Via config** (can be overridden by `--enable-llm` flag):
   ```json
   {
     "memory": {
       "enableLlmFallback": false
     }
   }
   ```

When LLM is disabled:
- Agent uses only cached/memory-based responses
- Returns deterministic "UNKNOWN" when memory cannot answer
- No API calls to LLM providers (saves cost and ensures privacy)
- Ideal for testing or offline operation

### 7. Gateway Configuration

Configure the HTTP server:

```json
{
  "gateway": {
    "host": "0.0.0.0",
    "port": 18790
  }
}
```

## Multi-Provider Setup

Nanobot automatically selects the right provider based on the model name:

**Example 1: Using OpenAI**
```json
{
  "agents": {
    "defaults": {
      "model": "gpt-4"
    }
  },
  "providers": {
    "openai": {
      "apiKey": "sk-..."
    }
  }
}
```

**Example 2: Using Gemini**
```json
{
  "agents": {
    "defaults": {
      "model": "gemini/gemini-2.0-flash-exp"
    }
  },
  "providers": {
    "gemini": {
      "apiKey": "..."
    }
  }
}
```

**Example 3: Multiple Providers**
```json
{
  "providers": {
    "openai": {
      "apiKey": "sk-..."
    },
    "gemini": {
      "apiKey": "..."
    },
    "anthropic": {
      "apiKey": "sk-ant-..."
    }
  }
}
```

Then switch models by changing `agents.defaults.model`:
- `"gpt-4"` → uses OpenAI
- `"gemini/gemini-2.0-flash-exp"` → uses Gemini
- `"claude-opus-4"` → uses Anthropic

## Common Issues

### "Extra inputs are not permitted" Error

This error occurs when your config.json contains fields that are not defined in the schema.

**Solution:**
1. Compare your config.json with the example: `config.example.json`
2. Remove any fields not present in the example
3. Make sure field names use camelCase (e.g., `apiKey` not `api_key`)

**Common culprits:**
- Old memory fields like `memoryType`, `vectorStore`, `embeddingProvider`
- Typos in field names
- Extra custom fields

### No API Key Configured

**Solution:**
1. Open `~/.nanobot/config.json`
2. Find the provider you want to use (e.g., `providers.openai`)
3. Replace the placeholder with your actual API key
4. Make sure the `model` in `agents.defaults` matches your provider

### Memory Not Working

**Checklist:**
- [ ] `memory.enabled` is `true`
- [ ] `memory.provider` is `"local"` (or `"mem0"` with valid API key)
- [ ] Workspace directory exists and is writable
- [ ] For embeddings: OpenAI API key is configured
- [ ] For mem0: `mem0ApiKey` is set

## Environment Variables

You can also configure Nanobot using environment variables with the prefix `NANOBOT_`:

```bash
# Set OpenAI API key
export NANOBOT_PROVIDERS__OPENAI__API_KEY="sk-..."

# Set model
export NANOBOT_AGENTS__DEFAULTS__MODEL="gpt-4"

# Enable memory
export NANOBOT_MEMORY__ENABLED="true"
```

Note the double underscore `__` for nested fields.

## Validation

To validate your config:

```python
from nanobot.config.loader import load_config

try:
    config = load_config()
    print("✓ Config is valid!")
    print(f"Using model: {config.agents.defaults.model}")
    print(f"Memory enabled: {config.memory.enabled}")
except Exception as e:
    print(f"✗ Config error: {e}")
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive values in production
3. **Restrict file access** with `tools.restrictToWorkspace: true`
4. **Use allowlists** in channel configs to control access
5. **Keep config.json permissions secure**: `chmod 600 ~/.nanobot/config.json`

## See Also

- [README.md](README.md) - Main documentation
- [nanobot/config/schema.py](nanobot/config/schema.py) - Complete schema definition
- [nanobot/providers/registry.py](nanobot/providers/registry.py) - Provider registry

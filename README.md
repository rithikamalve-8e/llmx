# llmx

A lightweight, provider-agnostic Python LLM client.
Supports OpenAI, Groq, and Gemini behind a single interface.

---

## Installation

```bash
git clone https://github.com/rithikamalve-8e/llmx.git
cd llmx
pip install -e .
```

---

## Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
```

---

## Quick Start

```python
from dotenv import load_dotenv
load_dotenv()

from llmx import LLMClient

client = LLMClient()
response = client.generate("Hello!")
print(response.content)
```

---

## Core Concepts

### LLMClient

The main entry point. Handles provider selection and request routing.

### Message

Represents a single chat message:

```python
Message(role="user", content="Hello")
```

### GenerateRequest / GenerateResponse

Standardized request/response objects used across all providers.

### BaseProvider

Abstract class that all providers must implement:

* `generate()`
* `stream()`

---

## Provider Selection

### Explicit Provider

```python
client = LLMClient(provider="openai")
```

### Auto-detection

If no provider is specified:

* The client checks environment variables
* Picks the first matching provider

### Priority

```
Explicit provider > Auto-detection
```

---

## Usage

### Simple Prompt

```python
response = client.generate("What is the capital of France?")
print(response.content)
```

---

### With Parameters

```python
response = client.generate(
    "Write a sentence about oxygen",
    model="gpt-4o",
    temperature=0.9,
    max_tokens=100,
)
```

---

### Streaming

```python
for chunk in client.stream("Count from 1 to 5"):
    print(chunk.delta, end="", flush=True)
```

---

### Switching Providers

```python
gemini_client = client.use("gemini")
response = gemini_client.generate("Hello from Gemini!")
```

---

## Tool Calling

Define tools:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
        },
    }
]
```

Use:

```python
response = client.generate("What's the weather in Tokyo?", tools=tools)

if response.tool_calls:
    tc = response.tool_calls[0]
    print(tc.name, tc.arguments)
```

---

## Error Handling

Common errors:

* Missing API key → set `.env`
* Unknown provider → check registry
* Invalid prompt type → must be `str`, `list[Message]`, or `GenerateRequest`

---

## Adding a New Provider

To add a new provider (example: Anthropic):

1. Create a new file:

```
llmx/providers/anthropic.py
```

2. Implement:

```python
class AnthropicProvider(BaseProvider):
    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def generate(self, request):
        ...

    def stream(self, request):
        ...
```

3. Register it in:

```
providers/__init__.py
```

```python
'anthropic': ('llmx.providers.anthropic', 'AnthropicProvider')
```

That’s it — it can now be used via:

```python
LLMClient(provider="anthropic")
```

---

## Project Structure

```
llmx/
  core.py
  models.py
  providers/
    base.py
    openai.py
    groq.py
    gemini.py
```

---

## Design Philosophy

* Provider-agnostic interface
* Minimal abstraction
* Easy to extend
* No vendor lock-in

---

## Limitations

* No automatic fallback if a provider fails
* Model defaults vary by provider
* Tool calling support depends on provider

---

## Roadmap

* Provider fallback support
* Configurable provider priority
* Improved logging

---

## Summary

`llmx` provides a clean abstraction over multiple LLM providers, allowing you to write code once and switch providers seamlessly.

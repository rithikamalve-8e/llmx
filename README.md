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

## Usage

```python
from dotenv import load_dotenv
load_dotenv()

from llmx import LLMClient

client = LLMClient()
response = client.generate("Hello!")
print(response.content)
```

import os
from dotenv import load_dotenv
load_dotenv()
from llmx import LLMClient, Message

client = LLMClient()  
print(client)         


# --- Simple string prompt ---
response = client.generate("What is the capital of France?", model="gpt-4o")
print(response.content)
print(response.usage)


# --- With options ---
response = client.generate(
    "Write a sentence about oxygen",
    model="gpt-4o",
    temperature=0.9,
    max_tokens=100,
)
print(response.content)


# --- Multi-turn conversation ---
messages = [
    Message(role="user", content="My name is Rithika."),
    Message(role="assistant", content="Nice to meet you, Rithika!"),
    Message(role="user", content="What's my name?"),
]
response = client.generate(messages, system="You are a helpful assistant.")
print(response.content)


# --- Streaming ---
for chunk in client.stream("Count from 1 to 5 slowly."):
    print(chunk.delta, end="", flush=True)
print()


# --- Explicit provider ---
groq_client = LLMClient(provider="gemini")
response = groq_client.generate("Hello from Groq!")
print(response.content)


# --- Switch provider on the fly ---
gemini_client = client.use("gemini")
response = gemini_client.generate("Hello from Gemini!")
print(response.content)


# --- Tool calling (OpenAI / Groq) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
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
response = client.generate("What's the weather in Tokyo?", tools=tools)
if response.tool_calls:
    tc = response.tool_calls[0]
    print(f"Tool: {tc.name}, Args: {tc.arguments}")
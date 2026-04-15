import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llmx import LLMClient, Message

# ------------------------------------------------------------------
# Validate environment
# ------------------------------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set in environment")

# ------------------------------------------------------------------
# Initialize client
# ------------------------------------------------------------------
client = LLMClient(provider="openai")
print(f"Client initialized: {client}")

# ------------------------------------------------------------------
# Simple string prompt
# ------------------------------------------------------------------
try:
    response = client.generate(
        "What is the capital of France?",
        model="gpt-4o"
    )
    print("\n[Simple Prompt]")
    print("Response:", response.content)
    print("Usage:", response.usage)
except Exception as e:
    print("Error in simple prompt:", e)

# ------------------------------------------------------------------
# With options
# ------------------------------------------------------------------
try:
    response = client.generate(
        "Write a sentence about oxygen",
        model="gpt-4o",
        temperature=0.9,
        max_tokens=100,
    )
    print("\n[With Options]")
    print(response.content)
except Exception as e:
    print("Error in options call:", e)

# ------------------------------------------------------------------
# Multi-turn conversation
# ------------------------------------------------------------------
try:
    messages = [
        Message(role="user", content="My name is Rithika."),
        Message(role="assistant", content="Nice to meet you, Rithika!"),
        Message(role="user", content="What's my name?"),
    ]

    response = client.generate(
        messages,
        model="gpt-4o",
        system="You are a helpful assistant."
    )

    print("\n[Multi-turn Conversation]")
    print(response.content)
except Exception as e:
    print("Error in conversation:", e)

# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------
try:
    print("\n[Streaming]")
    for chunk in client.stream(
        "Count from 1 to 5 slowly.",
        model="gpt-4o"
    ):
        print(chunk.delta or "", end="", flush=True)
    print()
except Exception as e:
    print("Error in streaming:", e)

# ------------------------------------------------------------------
# Explicit provider (Groq)
# ------------------------------------------------------------------
try:
    groq_client = LLMClient(provider="groq")

    response = groq_client.generate(
        "Hello from Groq!",
        model="groq/compound-mini"
    )

    print("\n[Groq]")
    print(response.content)
except Exception as e:
    print("Error with Groq:", e)

# ------------------------------------------------------------------
# Switch provider dynamically
# ------------------------------------------------------------------
try:
    gemini_client = client.use("gemini")

    response = gemini_client.generate(
        "Hello from Gemini!",
        model="gemini-2.5-flash"
    )

    print("\n[Gemini]")
    print(response.content)
except Exception as e:
    print("Error switching to Gemini:", e)

# ------------------------------------------------------------------
# Tool calling
# ------------------------------------------------------------------
try:
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

    response = client.generate(
        "What's the weather in Tokyo?",
        model="gpt-4o",
        tools=tools
    )

    print("\n[Tool Calling]")

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool: {tc.name}, Args: {tc.arguments}")
    else:
        print("No tool calls returned")
        print("Response:", response.content)

except Exception as e:
    print("Error in tool calling:", e)
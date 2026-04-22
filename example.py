import os
import asyncio
from llmx import LLMClient, Message
from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def log(title):
    print(f"\n{'='*10} {title} {'='*10}")

def ok(msg):
    print(f"[OK] {msg}")

def fail(msg, e=None):
    print(f"[FAIL] {msg}")
    if e:
        print("   →", repr(e))


# ------------------------------------------------------------------
# ENV CHECK
# ------------------------------------------------------------------

OPENAI = bool(os.getenv("OPENAI_API_KEY"))
GROQ = bool(os.getenv("GROQ_API_KEY"))
GEMINI = bool(os.getenv("GEMINI_API_KEY"))

print("Env status:")
print("OPENAI:", OPENAI)
print("GROQ:", GROQ)
print("GEMINI:", GEMINI)


# ------------------------------------------------------------------
# OPENAI TESTS
# ------------------------------------------------------------------

if OPENAI:
    log("OpenAI Integration")

    client = LLMClient(provider="openai")

    # Simple prompt
    try:
        resp = client.generate("Say hello world", model="gpt-4o")
        print("Response:", resp.content)
        ok("Simple prompt works")
    except Exception as e:
        fail("Simple prompt failed", e)

    # With options
    try:
        resp = client.generate(
            "Write one short sentence",
            model="gpt-4o",
            temperature=0.9,
            max_tokens=50,
        )
        print("Response:", resp.content)
        ok("Options work")
    except Exception as e:
        fail("Options test failed", e)

    # Multi-turn
    try:
        msgs = [
            Message("user", "My name is Rithika"),
            Message("assistant", "Nice to meet you"),
            Message("user", "What's my name?"),
        ]
        resp = client.generate(msgs, model="gpt-4o")
        print("Response:", resp.content)

        if "Rithika" in resp.content:
            ok("Multi-turn works")
        else:
            fail("Multi-turn incorrect response")

    except Exception as e:
        fail("Multi-turn failed", e)

    # Streaming
    try:
        print("Streaming:")
        chunks = []
        for c in client.stream("Count to 3", model="gpt-4o"):
            print(c.delta or "", end="", flush=True)
            chunks.append(c)

        print()
        if len(chunks) > 0:
            ok("Streaming works")
        else:
            fail("No streaming output")

    except Exception as e:
        fail("Streaming failed", e)

    # Tool calling
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        resp = client.generate(
            "What's the weather in Tokyo?",
            model="gpt-4o",
            tools=tools,
        )

        if resp.tool_calls:
            print("Tool calls:", resp.tool_calls)
            ok("Tool calling works")
        else:
            fail("No tool calls returned")

    except Exception as e:
        fail("Tool calling failed", e)


# ------------------------------------------------------------------
# GROQ TESTS
# ------------------------------------------------------------------

if GROQ:
    log("Groq Integration")

    client = LLMClient(provider="groq")

    try:
        resp = client.generate("Say hello", model="groq/compound-mini")
        print("Response:", resp.content)
        ok("Groq generate works")
    except Exception as e:
        fail("Groq generate failed", e)

    try:
        print("Streaming:")
        chunks = list(client.stream("Count to 3", model="groq/compound-mini"))
        print("Chunks:", len(chunks))
        ok("Groq streaming works")
    except Exception as e:
        fail("Groq streaming failed", e)


# ------------------------------------------------------------------
# GEMINI TESTS
# ------------------------------------------------------------------

if GEMINI:
    log("Gemini Integration")

    client = LLMClient(provider="gemini")

    try:
        resp = client.generate("Say hello", model="gemini-2.5-flash")
        print("Response:", resp.content)
        ok("Gemini generate works")
    except Exception as e:
        fail("Gemini generate failed", e)

    try:
        chunks = list(client.stream("Count to 3", model="gemini-2.5-flash"))
        print("Chunks:", len(chunks))
        ok("Gemini streaming works")
    except Exception as e:
        fail("Gemini streaming failed", e)


# ------------------------------------------------------------------
# CLIENT BEHAVIOR
# ------------------------------------------------------------------

log("Client Behavior")

# Provider switching
try:
    client = LLMClient(provider="openai")

    if GROQ:
        new_client = client.use("groq")
        print("Old:", client.provider_name)
        print("New:", new_client.provider_name)

        if new_client is not client:
            ok("Provider switching works")
        else:
            fail("Client reused incorrectly")

except Exception as e:
    fail("Provider switching failed", e)


# Sync vs async
try:
    client = LLMClient(provider="openai")

    sync_resp = client.generate("Say hi", model="gpt-4o")

    async def run():
        return await client.agenerate("Say hi", model="gpt-4o")

    async_resp = asyncio.run(run())

    print("Sync:", sync_resp.content)
    print("Async:", async_resp.content)

    ok("Sync vs Async works")

except Exception as e:
    fail("Sync vs Async failed", e)
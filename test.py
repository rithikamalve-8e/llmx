from llmx import LLMClient
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
load_dotenv()

client=LLMClient()
response=client.generate("hello how are you, is your name gemini", model="groq/compound-mini")
print(response.content)
from llmx import LLMClient
from dotenv import load_dotenv
load_dotenv()

client=LLMClient()
response=client.generate("hello how are you, is your name gemini", model="groq/compound-mini")

openai_client= client.use("openai")
response1=openai_client.generate("hello how are you, is your name gemini", model="gpt-4o")

print(response.content)
print(response1.content)
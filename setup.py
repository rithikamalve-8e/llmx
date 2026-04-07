from setuptools import setup, find_packages

setup(
    name="llmx",
    version="0.1.0",
    description="An LLM client with multiple provider support",
    author="Rithika",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "openai":  ["openai"],
        "groq":    ["groq"],
        "gemini":  ["google-generativeai"],
        "all":     ["openai", "groq", "google-generativeai"],
    },
    include_package_data=True,
)
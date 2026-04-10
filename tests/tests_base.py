import pytest
from llmx.providers.base import BaseProvider
from llmx.models import GenerateRequest


# -----------------------
# Dummy subclass for testing ABC
# -----------------------
class DummyProvider(BaseProvider):
    async def generate(self, request):
        return "ok"

    async def stream(self, request):
        if False:
            yield


def test_baseprovider_instantiation():
    p = DummyProvider()
    assert p.name == "base"


def test_build_messages_with_system():
    req = GenerateRequest(
        messages=[],
        system="sys prompt"
    )

    p = DummyProvider()
    msgs = p._build_messages(req)

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys prompt"


def test_build_messages_without_system():
    req = GenerateRequest(messages=[])

    p = DummyProvider()
    msgs = p._build_messages(req)

    assert msgs == []
from llmx.providers.base import BaseProvider
from llmx.models import GenerateRequest, Message


class DummyProvider(BaseProvider):
    def generate(self, request):
        pass

    def stream(self, request):
        pass


def test_build_messages_with_system():
    provider = DummyProvider()

    req = GenerateRequest(
        messages=[Message(role="user", content="Hi")],
        system="You are helpful"
    )

    msgs = provider._build_messages(req)

    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_messages_without_system():
    provider = DummyProvider()

    req = GenerateRequest(
        messages=[Message(role="user", content="Hi")]
    )

    msgs = provider._build_messages(req)

    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
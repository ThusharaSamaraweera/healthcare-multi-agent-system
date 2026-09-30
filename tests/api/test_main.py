from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

import api.main as main


class FakeAgent:
    def __init__(self, answer: str = "Hello from the API"):
        self.answer = answer
        self.calls = []

    async def achat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            answer=self.answer,
            route="direct",
            routes=["direct"],
            action=None,
            tool_output="",
            memory_context="",
            latency_ms=3,
        )


@pytest.fixture
def fake_agent():
    agent = FakeAgent()
    main.app.state.agent = agent
    yield agent
    main.app.state.agent = None


async def request(method: str, path: str, **kwargs):
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_health_reports_ok_when_agent_is_available(fake_agent):
    response = await request("GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_reports_starting_when_agent_is_unavailable():
    main.app.state.agent = None

    response = await request("GET", "/health")

    assert response.status_code == 200
    assert response.json()["status"] == "starting"


@pytest.mark.asyncio
async def test_chat_calls_agent_achat_and_maps_response(fake_agent):
    payload = {
        "user_message": "Hello",
        "user_id": "demo-user",
        "session_id": "test-session",
    }

    response = await request("POST", "/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "Hello from the API"
    assert response.json()["route"] == "direct"
    assert fake_agent.calls == [payload]


@pytest.mark.asyncio
async def test_chat_returns_503_when_agent_is_unavailable():
    main.app.state.agent = None

    response = await request(
        "POST",
        "/chat",
        json={
            "user_message": "Hello",
            "user_id": "demo-user",
            "session_id": "test-session",
        },
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_rejects_invalid_request():
    response = await request(
        "POST",
        "/chat",
        json={"user_message": "", "user_id": "demo-user"},
    )

    assert response.status_code == 422


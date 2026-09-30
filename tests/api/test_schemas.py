import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatResponse, HealthResponse


def test_chat_request_requires_non_empty_user_message_and_user_id():
    request = ChatRequest(
        user_message="Hello",
        user_id="demo-user",
    )

    assert request.session_id == "default"


@pytest.mark.parametrize(
    "payload",
    [
        {"user_message": "", "user_id": "demo-user"},
        {"user_message": "Hello", "user_id": ""},
        {"user_message": "Hello"},
    ],
)
def test_chat_request_rejects_invalid_payloads(payload):
    with pytest.raises(ValidationError):
        ChatRequest(**payload)


def test_chat_response_defaults_optional_fields():
    response = ChatResponse(answer="Hello", route="direct")

    assert response.routes == []
    assert response.action is None
    assert response.tool_output == ""
    assert response.memory_context == ""
    assert response.latency_ms == 0


def test_health_response_defaults_to_ok_with_tools_disabled():
    health = HealthResponse()

    assert health.status == "ok"
    assert health.tools.crm is False
    assert health.tools.rag is False
    assert health.tools.web_search is False

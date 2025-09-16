import json
from pathlib import Path
from jsonschema import validate

def load_schema():
    schema_path = Path("protocols/message.schema.json")
    assert schema_path.exists(), "missing message schema"
    return json.loads(schema_path.read_text())

def test_message_example_validates():
    schema = load_schema()
    example = {
        "id":"msg-1",
        "parent_id": None,
        "timestamp":"2025-09-16T18:30:00Z",
        "role":"planner",
        "intent":"plan",
        "content":"Break down task into steps.",
        "context":{"topic":"FRM retrieval eval","task_id":"t-123","session_id":"s-xyz"},
        "constraints":{"deadline_s":20,"max_tokens":512,"safety":["no_pii"]},
        "tools_requested":[{"name":"memory.search","args":{"query":"FRM retrieval eval","top_k":6}}],
        "citations":[],
        "privacy":"internal",
        "trace_id":"trace-abc"
    }
    validate(instance=example, schema=schema)

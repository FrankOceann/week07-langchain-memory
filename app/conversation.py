import json


def build_conversation_key(user_id: str, session_id: str) -> str:
    """Return an unambiguous key for user-scoped conversation storage."""
    return json.dumps(
        [user_id, session_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_workflow_thread_id(user_id: str, session_id: str) -> str:
    return f"week08:{build_conversation_key(user_id, session_id)}"

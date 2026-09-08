from app.conversation import build_conversation_key, build_workflow_thread_id


def test_conversation_scope_keys_are_unambiguous_for_colon_inputs():
    first = ("a:b", "c")
    second = ("a", "b:c")

    assert build_conversation_key(*first) != build_conversation_key(*second)
    assert build_workflow_thread_id(*first) != (
        build_workflow_thread_id(*second)
    )

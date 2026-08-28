import pytest

from main import build_parser


def test_chat_command_requires_session_id_and_user_id():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--session-id", "redis-demo"])

    args = parser.parse_args(
        ["chat", "--session-id", "redis-demo", "--user-id", "frank"]
    )

    assert (args.command, args.session_id, args.user_id) == (
        "chat",
        "redis-demo",
        "frank",
    )


def test_memory_add_requires_valid_category():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "memory",
                "add",
                "--user-id",
                "frank",
                "--category",
                "unknown",
                "--content",
                "内容",
            ]
        )

def test_memory_commands_parse_expected_arguments():
    parser = build_parser()

    add = parser.parse_args(
        [
            "memory",
            "add",
            "--user-id",
            "frank",
            "--category",
            "preference",
            "--content",
            "使用中文",
        ]
    )
    deactivate = parser.parse_args(
        ["memory", "deactivate", "--memory-id", "101"]
    )

    assert (add.command, add.memory_command, add.user_id) == (
        "memory",
        "add",
        "frank",
    )
    assert deactivate.memory_id == 101
from app.tools import InMemoryNotificationTool


def test_in_memory_notification_tool_records_sent_message():
    tool = InMemoryNotificationTool()

    result = tool.send("项目测试已经完成")

    assert result == "模拟通知已发送：项目测试已经完成"
    assert tool.sent_messages == ["项目测试已经完成"]

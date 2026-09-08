class InMemoryNotificationTool:
    def __init__(self):
        self.sent_messages = []

    def send(self, message: str) -> str:
        self.sent_messages.append(message)
        return f"模拟通知已发送：{message}"

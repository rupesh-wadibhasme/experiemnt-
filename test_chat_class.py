import unittest
from unittest.mock import MagicMock
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.chatbot import Chat


class TestChat(unittest.TestCase):
    def setUp(self):
        """
        Set up the Chat instance with in-memory storage.
        """
        history = InMemoryChatMessageHistory()
        self.chat = Chat(history)

    def test_empty(self):
        """
        Test the `empty` property.
        """
        self.assertTrue(self.chat.empty)
        self.chat.add("user", "Hello")
        self.assertFalse(self.chat.empty)

    def test_messages(self):
        """
        Test the `messages` property.
        """
        self.chat.add("system", "Welcome!")
        self.chat.add("user", "Hello")
        self.chat.add("assistant", "Hi, how can I help you?")
        self.chat.add("custom_role", "I am a custom role")

        expected = [
            {"role": "system", "content": "Welcome!"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
            {"role": "custom_role", "content": "I am a custom role"}
        ]

        self.assertEqual(self.chat.messages, expected)

    def test_add(self):
        """
        Test the `add` method.
        """
        message = self.chat.add("user", "Hello")
        self.assertEqual(message.content, "Hello")
        self.assertIsInstance(message, HumanMessage)
        self.assertEqual(len(self.chat.history.messages), 1)


if __name__ == "__main__":
    unittest.main()

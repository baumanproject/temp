from __future__ import annotations

from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output
from lfx.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_USER


class MinimalChatHistoryAdapter(Component):
    display_name = "Minimal Chat History Adapter"
    description = "Converts history rows to chat history for Tool Calling Agent."
    name = "MinimalChatHistoryAdapter"
    icon = "messages-square"

    inputs = [
        HandleInput(
            name="history",
            display_name="History",
            input_types=["DataFrame", "Table", "Data", "Message"],
            required=True,
        ),
    ]

    outputs = [
        Output(
            name="chat_history",
            display_name="Chat History",
            method="build_chat_history",
        ),
    ]

    def build_chat_history(self) -> list[Data]:
        rows = self._to_rows(self.history)

        messages: list[Message] = []

        for row in rows:
            text = row.get("text") or row.get("message") or row.get("content")

            if not text:
                continue

            sender = self._normalize_sender(
                row.get("sender") or row.get("role") or row.get("sender_name")
            )

            messages.append(
                Message(
                    text=str(text),
                    sender=sender,
                    sender_name="User" if sender == MESSAGE_SENDER_USER else "AI",
                    session_id=str(row.get("session_id") or ""),
                )
            )

        self.status = {
            "input_rows": len(rows),
            "output_messages": len(messages),
        }

        return messages

    @staticmethod
    def _normalize_sender(value: Any) -> str:
        sender = str(value or "").strip().lower()

        if sender in {"user", "human"}:
            return MESSAGE_SENDER_USER

        return MESSAGE_SENDER_AI

    @staticmethod
    def _to_rows(history: Any) -> list[dict[str, Any]]:
        if history is None:
            return []

        if isinstance(history, list):
            result = []

            for item in history:
                if isinstance(item, Message):
                    result.append(item.data)
                elif isinstance(item, Data):
                    result.append(item.data)
                elif isinstance(item, dict):
                    result.append(item)

            return result

        if isinstance(history, Message):
            return [history.data]

        if isinstance(history, Data):
            data = history.data

            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]

            if isinstance(data, dict):
                return [data]

        if hasattr(history, "to_dict"):
            try:
                return history.to_dict(orient="records")
            except TypeError:
                return []

        return []

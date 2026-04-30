from __future__ import annotations

from typing import Any

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output


class MinimalChatHistoryAdapter(Component):
    display_name = "Minimal Chat History Adapter"
    description = "Converts history rows to Message list for Tool Calling Agent Chat Memory."
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
            types=["Data"],
            selected="Data",
        ),
    ]

    def build_chat_history(self) -> list[Data]:
        rows = self._to_rows(self.history)

        messages: list[Message] = []

        for row in rows:
            clean_row = self._unwrap_row(row)

            text = (
                clean_row.get("text")
                or clean_row.get("message")
                or clean_row.get("content")
            )

            if not text:
                continue

            sender = self._normalize_sender(
                clean_row.get("sender")
                or clean_row.get("role")
                or clean_row.get("sender_name")
            )

            messages.append(
                Message(
                    text=str(text),
                    sender=sender,
                    sender_name="User" if sender == "User" else "AI",
                    session_id=str(clean_row.get("session_id") or ""),
                )
            )

        self.log(
            f"Chat history adapter: input_rows={len(rows)}, output_messages={len(messages)}",
            name="chat_history_adapter_debug",
        )

        return messages

    @staticmethod
    def _normalize_sender(value: Any) -> str:
        sender = str(value or "").strip().lower()

        if sender in {"user", "human", "person"}:
            return "User"

        return "Machine"

    @staticmethod
    def _unwrap_row(row: Any) -> dict[str, Any]:
        if isinstance(row, Message):
            return row.data

        if isinstance(row, Data):
            return row.data

        if isinstance(row, dict):
            nested_data = row.get("data")

            if isinstance(nested_data, dict):
                return nested_data

            return row

        if hasattr(row, "data") and isinstance(row.data, dict):
            return row.data

        return {}

    @classmethod
    def _to_rows(cls, history: Any) -> list[dict[str, Any]]:
        if history is None:
            return []

        if isinstance(history, list):
            return [cls._unwrap_row(item) for item in history]

        if isinstance(history, Message):
            return [history.data]

        if isinstance(history, Data):
            data = history.data

            if isinstance(data, list):
                return [cls._unwrap_row(item) for item in data]

            if isinstance(data, dict):
                return [cls._unwrap_row(data)]

        if hasattr(history, "to_dict"):
            try:
                records = history.to_dict(orient="records")

                if isinstance(records, list):
                    return [cls._unwrap_row(item) for item in records]
            except TypeError:
                return []

        if hasattr(history, "data"):
            data = history.data

            if isinstance(data, list):
                return [cls._unwrap_row(item) for item in data]

            if isinstance(data, dict):
                return [cls._unwrap_row(data)]

        return []

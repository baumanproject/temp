import json
import os
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, RootModel, model_validator


# ---------------------------
# Config
# ---------------------------

class AgentConfig(BaseModel):
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "CHANGE_ME"

    # Optional: Virtual key header for LiteLLM Proxy
    litellm_virtual_key: str | None = None
    litellm_virtual_key_header: str = "X-Litellm-Key"

    model: str = "gigachat"
    temperature: float = 0.0
    max_output_tokens: int = 1200

    # Attempts
    max_total_calls: int = 3
    max_fix_calls: int = 1  # сколько раз пробуем "починить" без исходного документа

    # Tracing
    trace_enabled: bool = True
    trace_print: bool = False
    trace_max_chars: int = 4000


class TokenUsageSummary(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


# ---------------------------
# Tracing
# ---------------------------

class TraceEvent(BaseModel):
    index: int
    purpose: str
    messages: list[dict[str, str]]
    tools_count: int
    tool_choice: dict | None
    assistant_content: str | None = None
    tool_arguments_raw: str | None = None
    parsed_result: dict[str, str | None] | None = None
    usage: dict[str, int] | None = None
    error: str | None = None


class AgentTracer:
    def __init__(self, enabled: bool, do_print: bool, max_chars: int) -> None:
        self._enabled = enabled
        self._do_print = do_print
        self._max_chars = int(max_chars)
        self._events: list[TraceEvent] = []
        self._i = 0

    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def start(self, purpose: str, messages: list[dict], tools: list[dict], tool_choice: dict | None) -> int:
        if not self._enabled:
            return -1
        self._i += 1
        ev = TraceEvent(
            index=self._i,
            purpose=purpose,
            messages=[self._trim_msg(m) for m in messages],
            tools_count=len(tools),
            tool_choice=tool_choice,
        )
        self._events.append(ev)
        if self._do_print:
            print(f"\n[TRACE] CALL #{ev.index} purpose={purpose} tools={len(tools)} tool_choice={tool_choice}")
        return ev.index

    def finish(
        self,
        trace_id: int,
        assistant_content: str | None,
        tool_arguments_raw: str | None,
        parsed_result: dict[str, str | None] | None,
        usage: dict[str, int] | None,
        error: str | None,
    ) -> None:
        if not self._enabled or trace_id < 0:
            return
        ev = next((x for x in self._events if x.index == trace_id), None)
        if not ev:
            return
        ev.assistant_content = self._trim_text(assistant_content)
        ev.tool_arguments_raw = self._trim_text(tool_arguments_raw)
        ev.parsed_result = parsed_result
        ev.usage = usage
        ev.error = error
        if self._do_print:
            print(f"[TRACE] RESULT #{trace_id} usage={usage} error={error}")

    def _trim_text(self, s: str | None) -> str | None:
        if s is None:
            return None
        if len(s) <= self._max_chars:
            return s
        return s[: self._max_chars] + "…(truncated)"

    def _trim_msg(self, m: dict) -> dict[str, str]:
        return {"role": str(m.get("role", "")), "content": self._trim_text(str(m.get("content", ""))) or ""}


# ---------------------------
# Strict validation: keys must be exactly attributes
# ---------------------------

class StrictAttributesResult(RootModel[dict[str, str | None]]):
    """
    RootModel позволяет ключи словаря любыми строками (русские/цифры/что угодно).
    Проверку ключей делаем сами в model_validator.  [oai_citation:2‡docs.pydantic.dev](https://docs.pydantic.dev/latest/concepts/validators/?utm_source=chatgpt.com)
    """
    _allowed_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_keys_and_types(self):
        data = self.root
        allowed = set(self.__class__._allowed_keys)

        extra = set(data.keys()) - allowed
        if extra:
            raise ValueError(f"Лишние ключи: {sorted(extra)}")

        missing = allowed - set(data.keys())
        if missing:
            raise ValueError(f"Отсутствующие ключи: {sorted(missing)}")

        # типы: str|null; если модель дала число/булево — приводим к str
        for k, v in list(data.items()):
            if v is None:
                continue
            if not isinstance(v, str):
                data[k] = str(v)

        return self

    @classmethod
    def for_attributes(cls, attributes: list[str]) -> type["StrictAttributesResult"]:
        # создаём подкласс с заданным allowed list
        attrs = tuple(attributes)
        return type("StrictAttributesResultForTask", (cls,), {"_allowed_keys": attrs})


# ---------------------------
# Tool schema factory (NO attribute modifications)
# ---------------------------

class ToolSchemaFactory:
    """
    Генерируем tools schema вручную:
    - properties: ключи = атрибуты КАК ЕСТЬ
    - required: все атрибуты (ключи должны присутствовать, даже если null)
    - additionalProperties: false
    Это и есть включение tool calling: параметр tools + tool_choice.  [oai_citation:3‡platform.openai.com](https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com)
    """

    def __init__(self, tool_name: str = "extract_insurance_attributes") -> None:
        self._tool_name = tool_name

    def build(self, attributes: list[str]) -> tuple[list[dict], dict]:
        properties: dict[str, dict] = {}
        for a in attributes:
            properties[a] = {
                "description": f"Значение атрибута '{a}' из текста. Если не найдено — null.",
                "anyOf": [{"type": "string"}, {"type": "null"}],
            }

        tool = {
            "type": "function",
            "function": {
                "name": self._tool_name,
                "description": "Извлечь значения заданных атрибутов из описания страхового продукта.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(attributes),
                    "additionalProperties": False,
                },
            },
        }

        tool_choice = {"type": "function", "function": {"name": self._tool_name}}
        return [tool], tool_choice


# ---------------------------
# Agent
# ---------------------------

class InsuranceAttributeExtractionAgent:
    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._client = self._build_openai_client()
        self._tracer = AgentTracer(config.trace_enabled, config.trace_print, config.trace_max_chars)
        self._tool_factory = ToolSchemaFactory()

    def extract(
        self,
        markdown_text: str,
        attributes: list[str],
    ) -> tuple[dict[str, str | None] | None, TokenUsageSummary, list[TraceEvent]]:
        usage_sum = TokenUsageSummary()

        tools, tool_choice = self._tool_factory.build(attributes)
        ValidatorModel = StrictAttributesResult.for_attributes(attributes)

        # 1) Основной запрос (с жирным документом)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_payload(markdown_text, attributes)},
        ]

        calls_left = int(self._cfg.max_total_calls)
        fix_left = int(self._cfg.max_fix_calls)

        last_assistant_content: str | None = None
        last_tool_args_raw: str | None = None

        while calls_left > 0:
            calls_left -= 1

            purpose = "extract" if fix_left == self._cfg.max_fix_calls else "fix"
            trace_id = self._tracer.start(purpose, messages, tools, tool_choice)

            try:
                # >>> ВОТ ТУТ включён tool calling:
                #     tools=... и tool_choice=... в chat.completions.create  [oai_citation:4‡platform.openai.com](https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com)
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_output_tokens,
                )

                usage = self._extract_usage(resp)
                self._accumulate_usage(usage_sum, usage)

                msg = resp.choices[0].message
                last_assistant_content = getattr(msg, "content", None)

                tool_args_raw = self._extract_tool_arguments(msg)
                last_tool_args_raw = tool_args_raw

                if tool_args_raw is None:
                    raise ValueError("Модель не вернула tool_calls.function.arguments")

                data = json.loads(tool_args_raw)
                validated = ValidatorModel.model_validate(data)
                result = dict(validated.root)

                self._tracer.finish(
                    trace_id,
                    assistant_content=last_assistant_content,
                    tool_arguments_raw=tool_args_raw,
                    parsed_result=result,
                    usage=usage,
                    error=None,
                )
                return result, usage_sum, self._tracer.events()

            except Exception as e:
                usage = None
                # usage можем не получить, если ошибка случилась до resp/usage
                self._tracer.finish(
                    trace_id,
                    assistant_content=last_assistant_content,
                    tool_arguments_raw=last_tool_args_raw,
                    parsed_result=None,
                    usage=usage,
                    error=str(e),
                )

                # Если это первый провал — делаем "экономичный fix" без документа:
                if fix_left > 0 and calls_left > 0:
                    fix_left -= 1
                    messages = self._minimal_fix_messages(
                        attributes=attributes,
                        error=str(e),
                        last_tool_args_or_text=last_tool_args_raw or last_assistant_content or "",
                    )
                    continue

                break

        return None, usage_sum, self._tracer.events()

    def _build_openai_client(self) -> OpenAI:
        headers = {}
        if self._cfg.litellm_virtual_key:
            headers[self._cfg.litellm_virtual_key_header] = f"Bearer {self._cfg.litellm_virtual_key}"
            api_key = "not-used"
        else:
            api_key = self._cfg.litellm_api_key

        return OpenAI(
            base_url=self._cfg.litellm_base_url,
            api_key=api_key,
            default_headers=headers or None,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Ты извлекаешь значения атрибутов из текста страхового продукта.\n"
            "Всегда вызывай инструмент и возвращай аргументы инструмента.\n"
            "Не добавляй текстовых пояснений.\n"
            "Значения бери только из входного текста. Если не найдено — null.\n"
        )

    @staticmethod
    def _user_payload(markdown_text: str, attributes: list[str]) -> str:
        attrs = "\n".join(f"- {a}" for a in attributes)
        return (
            "Список атрибутов (ключи результата должны совпадать с ними ТОЧНО):\n"
            f"{attrs}\n\n"
            "Текст (markdown, включая таблицы):\n"
            "-----\n"
            f"{markdown_text}\n"
            "-----\n"
        )

    @staticmethod
    def _minimal_fix_messages(attributes: list[str], error: str, last_tool_args_or_text: str) -> list[dict]:
        """
        Экономия токенов: НЕ отправляем исходный markdown.
        Просим только пересобрать/исправить результат по списку атрибутов.
        """
        attrs = "\n".join(f"- {a}" for a in attributes)
        return [
            {
                "role": "system",
                "content": (
                    "Ты исправляешь результат, чтобы он строго соответствовал списку атрибутов.\n"
                    "Всегда вызывай инструмент и возвращай аргументы инструмента.\n"
                    "НЕ выдумывай новые значения. Только нормализуй структуру.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ошибка валидации: {error}\n\n"
                    f"Список атрибутов (ключи):\n{attrs}\n\n"
                    "Вот предыдущий результат (его надо исправить):\n"
                    "-----\n"
                    f"{last_tool_args_or_text}\n"
                    "-----\n"
                ),
            },
        ]

    @staticmethod
    def _extract_tool_arguments(message) -> str | None:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None
        fn = getattr(tool_calls[0], "function", None)
        if not fn:
            return None
        args = getattr(fn, "arguments", None)
        return str(args) if args is not None else None

    @staticmethod
    def _extract_usage(resp) -> dict[str, int] | None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None

        def g(key: str) -> int | None:
            if isinstance(usage, dict):
                v = usage.get(key)
                return int(v) if v is not None else None
            v = getattr(usage, key, None)
            return int(v) if v is not None else None

        inp = g("prompt_tokens") or g("input_tokens") or 0
        out = g("completion_tokens") or g("output_tokens") or 0
        total = g("total_tokens")
        if total is None:
            total = int(inp) + int(out)

        return {"input_tokens": int(inp), "output_tokens": int(out), "total_tokens": int(total)}

    @staticmethod
    def _accumulate_usage(total: TokenUsageSummary, usage: dict[str, int] | None) -> None:
        total.calls += 1
        if not usage:
            return
        total.input_tokens += usage["input_tokens"]
        total.output_tokens += usage["output_tokens"]
        total.total_tokens += usage["total_tokens"]


if __name__ == "__main__":
    cfg = AgentConfig(
        litellm_base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        litellm_api_key=os.getenv("LITELLM_API_KEY", "CHANGE_ME"),
        model=os.getenv("LLM_MODEL", "gigachat"),
        trace_enabled=os.getenv("TRACE_ENABLED", "1") == "1",
        trace_print=os.getenv("TRACE_PRINT", "0") == "1",
    )

    agent = InsuranceAttributeExtractionAgent(cfg)

    md_text = """
# Продукт “Супер-страхование”
| Параметр | Значение |
|---|---|
| Страховая сумма | 1 000 000 ₽ |
| Франшиза | 10 000 ₽ |
| Территория | РФ |
"""
    attrs = ["Страховая сумма", "Франшиза", "Территория", "Срок страхования 2025"]

    result, usage, trace = agent.extract(md_text, attrs)

    print("RESULT:", result)
    print("USAGE:", usage.model_dump())

    print("\nTRACE EVENTS:", len(trace))
    for ev in trace:
        print(f"\n--- CALL #{ev.index} purpose={ev.purpose} ---")
        print("tools_count:", ev.tools_count, "tool_choice:", ev.tool_choice)
        print("usage:", ev.usage)
        print("error:", ev.error)
        if ev.tool_arguments_raw:
            print("tool_arguments_raw:", ev.tool_arguments_raw[:300], "..." if len(ev.tool_arguments_raw) > 300 else "")

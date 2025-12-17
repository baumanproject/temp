# insurance_attribute_agent_tools.py

import json
import os
import time

from openai import OpenAI, DefaultHttpxClient
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, RootModel, model_validator


class AgentConfig(BaseModel):
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "CHANGE_ME"

    # Optional: Virtual key header for LiteLLM Proxy
    litellm_virtual_key: str | None = None
    litellm_virtual_key_header: str = "X-Litellm-Key"

    model: str = "gigachat"
    temperature: float = 0.0
    max_output_tokens: int = 1200

    # Сколько "смысловых" LLM-вызовов (extract + fix и т.д.)
    max_total_calls: int = 3

    # Fix без жирного документа: сколько раз можно попробовать
    max_fix_calls: int = 1

    # Retry по сети: если connection/timeout — повторяем запрос
    connection_retries: int = 1
    connection_retry_sleep_seconds: float = 0.5

    # Tracing
    trace_enabled: bool = True
    trace_print: bool = False
    trace_max_chars: int = 4000


class TokenUsageSummary(BaseModel):
    # все попытки (включая connection retries)
    attempts: int = 0
    responses_received: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TraceEvent(BaseModel):
    index: int
    purpose: str
    attempt_no: int
    messages: list[dict[str, str]]

    tools_count: int
    tool_choice: dict | None

    assistant_content: str | None = None
    tool_arguments_raw: str | None = None

    parsed_result: dict[str, str | None] | None = None
    usage: dict[str, int] | None = None

    error_kind: str | None = None   # "connection" | "validation" | "protocol" | "unknown"
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

    def start(
        self,
        purpose: str,
        attempt_no: int,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | None,
    ) -> int:
        if not self._enabled:
            return -1
        self._i += 1
        ev = TraceEvent(
            index=self._i,
            purpose=purpose,
            attempt_no=attempt_no,
            messages=[self._trim_msg(m) for m in messages],
            tools_count=len(tools),
            tool_choice=tool_choice,
        )
        self._events.append(ev)
        if self._do_print:
            print(f"\n[TRACE] CALL #{ev.index} purpose={purpose} attempt={attempt_no} tools={len(tools)}")
        return ev.index

    def finish(
        self,
        trace_id: int,
        assistant_content: str | None,
        tool_arguments_raw: str | None,
        parsed_result: dict[str, str | None] | None,
        usage: dict[str, int] | None,
        error_kind: str | None,
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
        ev.error_kind = error_kind
        ev.error = error

        if self._do_print:
            print(f"[TRACE] RESULT #{trace_id} error_kind={error_kind} error={error} usage={usage}")

    def _trim_text(self, s: str | None) -> str | None:
        if s is None:
            return None
        if len(s) <= self._max_chars:
            return s
        return s[: self._max_chars] + "…(truncated)"

    def _trim_msg(self, m: dict) -> dict[str, str]:
        return {"role": str(m.get("role", "")), "content": self._trim_text(str(m.get("content", ""))) or ""}


class StrictAttributesResult(RootModel[dict[str, str | None]]):
    """
    RootModel позволяет ключи любыми строками (русский/цифры/что угодно).
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

        for k, v in list(data.items()):
            if v is None:
                continue
            if not isinstance(v, str):
                data[k] = str(v)

        return self

    @classmethod
    def for_attributes(cls, attributes: list[str]) -> type["StrictAttributesResult"]:
        return type("StrictAttributesResultForTask", (cls,), {"_allowed_keys": tuple(attributes)})


class ToolSchemaFactory:
    """
    tools schema генерим вручную:
    - properties ключи = атрибуты как есть (ничего не модифицируем)
    - required = все атрибуты
    - additionalProperties=false
    Это и включает tool calling вместе с tool_choice.  [oai_citation:2‡docs.litellm.ai](https://docs.litellm.ai/docs/proxy/user_keys?utm_source=chatgpt.com)
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


class InsuranceAttributeExtractionAgent:
    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._client = self._build_openai_client()
        self._tools_factory = ToolSchemaFactory()
        self._tracer = AgentTracer(config.trace_enabled, config.trace_print, config.trace_max_chars)

    def extract(
        self,
        markdown_text: str,
        attributes: list[str],
    ) -> tuple[dict[str, str | None] | None, TokenUsageSummary, list[TraceEvent]]:
        usage_sum = TokenUsageSummary()

        tools, tool_choice = self._tools_factory.build(attributes)
        ValidatorModel = StrictAttributesResult.for_attributes(attributes)

        # 1) Основной запрос с документом
        messages = [
            {"role": "system", "content": self._system_prompt_extract()},
            {"role": "user", "content": self._user_payload(markdown_text, attributes)},
        ]

        calls_left = int(self._cfg.max_total_calls)
        fix_left = int(self._cfg.max_fix_calls)

        last_tool_args_or_text = ""

        while calls_left > 0:
            calls_left -= 1

            purpose = "extract" if fix_left == self._cfg.max_fix_calls else "fix"

            try:
                result, last_tool_args_or_text = self._call_with_connection_retry(
                    purpose=purpose,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    validator_model=ValidatorModel,
                    usage_sum=usage_sum,
                )
                return result, usage_sum, self._tracer.events()

            except _ConnectionFinalError as e:
                # Повторяли по сети, но не вышло — останавливаем пайплайн.
                return None, usage_sum, self._tracer.events()

            except Exception as e:
                # Любая валидация/протокол/JSON ошибка -> пробуем "тонкий fix" без документа
                if fix_left > 0 and calls_left > 0:
                    fix_left -= 1
                    messages = self._minimal_fix_messages(
                        attributes=attributes,
                        error=str(e),
                        last_tool_args_or_text=last_tool_args_or_text,
                    )
                    continue
                return None, usage_sum, self._tracer.events()

        return None, usage_sum, self._tracer.events()

    def _call_with_connection_retry(
        self,
        purpose: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict,
        validator_model: type[StrictAttributesResult],
        usage_sum: TokenUsageSummary,
    ) -> tuple[dict[str, str | None], str]:
        """
        Один "логический" вызов, внутри которого:
        - connection retries (APIConnectionError/APITimeoutError)  [oai_citation:3‡OpenAI платформы](https://platform.openai.com/docs/guides/error-codes?utm_source=chatgpt.com)
        - трассируем каждую попытку отдельно
        """
        last_tool_args_or_text = ""

        for attempt_no in range(1, int(self._cfg.connection_retries) + 2):
            trace_id = self._tracer.start(
                purpose=purpose,
                attempt_no=attempt_no,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            usage_sum.attempts += 1

            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_output_tokens,
                )

                usage = self._extract_usage(resp)
                if usage:
                    usage_sum.responses_received += 1
                    usage_sum.input_tokens += usage["input_tokens"]
                    usage_sum.output_tokens += usage["output_tokens"]
                    usage_sum.total_tokens += usage["total_tokens"]

                msg = resp.choices[0].message
                assistant_content = getattr(msg, "content", None)

                tool_args_raw = self._extract_tool_arguments(msg)
                if tool_args_raw is None:
                    last_tool_args_or_text = assistant_content or ""
                    self._tracer.finish(
                        trace_id,
                        assistant_content=assistant_content,
                        tool_arguments_raw=None,
                        parsed_result=None,
                        usage=usage,
                        error_kind="protocol",
                        error="Модель не вернула tool_calls.function.arguments",
                    )
                    raise ValueError("Модель не вернула tool_calls.function.arguments")

                last_tool_args_or_text = tool_args_raw

                data = json.loads(tool_args_raw)
                validated = validator_model.model_validate(data)
                result = dict(validated.root)

                self._tracer.finish(
                    trace_id,
                    assistant_content=assistant_content,
                    tool_arguments_raw=tool_args_raw,
                    parsed_result=result,
                    usage=usage,
                    error_kind=None,
                    error=None,
                )
                return result, last_tool_args_or_text

            except (APIConnectionError, APITimeoutError) as e:
                # connection error -> логируем и повторяем
                self._tracer.finish(
                    trace_id,
                    assistant_content=None,
                    tool_arguments_raw=None,
                    parsed_result=None,
                    usage=None,
                    error_kind="connection",
                    error=str(e),
                )
                if attempt_no <= int(self._cfg.connection_retries):
                    time.sleep(float(self._cfg.connection_retry_sleep_seconds))
                    continue
                raise _ConnectionFinalError(str(e)) from e

            except Exception as e:
                # любая другая ошибка (JSON/валидация/и т.д.)
                self._tracer.finish(
                    trace_id,
                    assistant_content=None,
                    tool_arguments_raw=last_tool_args_or_text or None,
                    parsed_result=None,
                    usage=None,
                    error_kind="validation",
                    error=str(e),
                )
                raise

        raise _ConnectionFinalError("Unexpected retry loop exit")

    def _build_openai_client(self) -> OpenAI:
        headers = {}
        if self._cfg.litellm_virtual_key:
            headers[self._cfg.litellm_virtual_key_header] = f"Bearer {self._cfg.litellm_virtual_key}"
            api_key = "not-used"
        else:
            api_key = self._cfg.litellm_api_key

        # ВНИМАНИЕ: verify=False отключает проверку TLS сертификата (опасно в проде).
        # Вы это просили явно.
        http_client = DefaultHttpxClient(verify=False)  #  [oai_citation:4‡Hugging Face](https://huggingface.co/koichi12/llm_tutorial/blob/04079614d32612f7824868db5cc8dfadd69fae63/.venv/lib/python3.11/site-packages/openai/_base_client.py?utm_source=chatgpt.com)

        return OpenAI(
            base_url=self._cfg.litellm_base_url,
            api_key=api_key,
            default_headers=headers or None,
            http_client=http_client,
            max_retries=0,  # чтобы retries были только нашими
        )

    @staticmethod
    def _system_prompt_extract() -> str:
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
        Экономия токенов: не шлём исходный markdown.
        Исправляем только структуру предыдущего результата.
        """
        attrs = "\n".join(f"- {a}" for a in attributes)
        return [
            {
                "role": "system",
                "content": (
                    "Ты исправляешь результат, чтобы он строго соответствовал списку атрибутов.\n"
                    "Всегда вызывай инструмент и возвращай аргументы инструмента.\n"
                    "НЕ выдумывай новые значения.\n"
                    "Если значения неизвестны — ставь null.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ошибка: {error}\n\n"
                    f"Список атрибутов:\n{attrs}\n\n"
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


class _ConnectionFinalError(RuntimeError):
    pass


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

    # Отладка: все обращения к модели (включая retries)
    print("\nTRACE EVENTS:", len(trace))
    for ev in trace:
        print(f"\n--- CALL #{ev.index} purpose={ev.purpose} attempt={ev.attempt_no} ---")
        print("error_kind:", ev.error_kind)
        print("error:", ev.error)
        print("usage:", ev.usage)
        if ev.tool_arguments_raw:
            print("tool_arguments_raw:", ev.tool_arguments_raw[:300], "..." if len(ev.tool_arguments_raw) > 300 else "")

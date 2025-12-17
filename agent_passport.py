# insurance_attribute_agent_tools.py

import os
import re

from openai import OpenAI

import instructor
from instructor.core.hooks import Hooks
from pydantic import BaseModel, Field, ConfigDict, create_model


class AgentConfig(BaseModel):
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "CHANGE_ME"

    # optional virtual key header for LiteLLM proxy
    litellm_virtual_key: str | None = None
    litellm_virtual_key_header: str = "X-Litellm-Key"

    model: str = "gigachat"
    temperature: float = 0.0
    max_output_tokens: int = 1200

    # attempts
    max_total_calls: int = 3
    max_format_fixes: int = 1

    # tracing
    trace_enabled: bool = True          # хранить трейс в памяти
    trace_print: bool = False           # печатать в stdout
    trace_max_chars: int = 4000         # обрезка длинных сообщений

    format_fix_user_prompt_template: str = (
        "Предыдущий ответ не прошёл валидацию.\n"
        "Ошибка (для ориентира): {error}\n\n"
        "Пересобери результат СТРОГО по списку атрибутов.\n"
        "Правила:\n"
        "1) Ключи — ТОЧНО как в списке атрибутов.\n"
        "2) Значения — строки или null.\n"
        "3) Запрещены лишние ключи.\n"
        "4) Верни ТОЛЬКО результат, без пояснений.\n"
    )


class TokenUsageSummary(BaseModel):
    calls_attempted: int = 0
    responses_received: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TokenLedger:
    """
    Hooks:
      - completion:kwargs -> фиксируем факт попытки + можно смотреть tools/tool_choice
      - completion:response -> usage + последний assistant text (часто это tool arguments)
     [oai_citation:4‡python.useinstructor.com](https://python.useinstructor.com/concepts/hooks/?utm_source=chatgpt.com)
    """
    def __init__(self) -> None:
        self._summary = TokenUsageSummary()
        self._last_assistant_text: str | None = None
        self._last_usage: dict[str, int] | None = None
        self._last_kwargs: dict | None = None

    def on_completion_kwargs(self, **kwargs) -> None:
        self._summary.calls_attempted += 1
        self._last_kwargs = kwargs

    def on_completion_response(self, response) -> None:
        self._summary.responses_received += 1
        self._last_assistant_text = self._extract_assistant_text(response)
        self._last_usage = self._extract_usage_dict(response)
        self._accumulate(self._last_usage)

    def summary(self) -> TokenUsageSummary:
        return self._summary

    def last_assistant_text(self) -> str | None:
        return self._last_assistant_text

    def last_usage(self) -> dict[str, int] | None:
        return self._last_usage

    def last_provider_kwargs(self) -> dict | None:
        return self._last_kwargs

    def _accumulate(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        self._summary.input_tokens += int(usage.get("input_tokens", 0))
        self._summary.output_tokens += int(usage.get("output_tokens", 0))
        self._summary.total_tokens += int(usage.get("total_tokens", 0))

    @staticmethod
    def _extract_usage_dict(response) -> dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        def get(key: str) -> int | None:
            if isinstance(usage, dict):
                v = usage.get(key)
                return int(v) if v is not None else None
            v = getattr(usage, key, None)
            return int(v) if v is not None else None

        prompt = get("prompt_tokens") or get("input_tokens") or 0
        comp = get("completion_tokens") or get("output_tokens") or 0
        total = get("total_tokens")
        if total is None:
            total = int(prompt) + int(comp)

        return {"input_tokens": int(prompt), "output_tokens": int(comp), "total_tokens": int(total)}

    @staticmethod
    def _extract_assistant_text(response) -> str | None:
        """
        В tool calling полезная нагрузка обычно в tool_calls[0].function.arguments.
        Это удобно для "тонкого" fix без исходного документа.  [oai_citation:5‡platform.openai.com](https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com)
        """
        try:
            msg = response.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                fn = getattr(tool_calls[0], "function", None)
                if fn:
                    args = getattr(fn, "arguments", None)
                    if args:
                        return str(args)

            content = getattr(msg, "content", None)
            if content:
                return str(content)

            refusal = getattr(msg, "refusal", None)
            if refusal:
                return str(refusal)
        except Exception:
            return None
        return None


class TraceEvent(BaseModel):
    index: int
    purpose: str
    request_messages: list[dict[str, str]]
    provider_kwargs_compact: dict | None = None
    assistant_text: str | None = None
    parsed_result: dict[str, str | None] | None = None
    usage: dict[str, int] | None = None
    error: str | None = None


class AgentTracer:
    def __init__(self, enabled: bool, do_print: bool, max_chars: int) -> None:
        self._enabled = enabled
        self._do_print = do_print
        self._max_chars = int(max_chars)
        self._events: list[TraceEvent] = []
        self._counter = 0

    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def start_call(self, purpose: str, messages: list[dict]) -> int:
        if not self._enabled:
            return -1
        self._counter += 1
        idx = self._counter
        ev = TraceEvent(
            index=idx,
            purpose=purpose,
            request_messages=[self._trim_msg(m) for m in messages],
        )
        self._events.append(ev)

        if self._do_print:
            print(f"\n[TRACE] CALL #{idx} purpose={purpose}")
        return idx

    def finish_call(
        self,
        trace_id: int,
        assistant_text: str | None,
        usage: dict[str, int] | None,
        parsed_result: dict[str, str | None] | None,
        error: str | None,
        provider_kwargs: dict | None,
    ) -> None:
        if not self._enabled or trace_id < 0:
            return
        ev = next((e for e in self._events if e.index == trace_id), None)
        if ev is None:
            return

        ev.assistant_text = self._trim_text(assistant_text)
        ev.usage = usage
        ev.parsed_result = parsed_result
        ev.error = error
        ev.provider_kwargs_compact = self._compact_kwargs(provider_kwargs)

        if self._do_print:
            print(f"[TRACE] RESULT #{trace_id} usage={usage} error={error}")

    def _trim_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars] + "…(truncated)"

    def _trim_msg(self, msg: dict) -> dict[str, str]:
        return {
            "role": str(msg.get("role", "")),
            "content": self._trim_text(str(msg.get("content", ""))) or "",
        }

    @staticmethod
    def _compact_kwargs(kwargs: dict | None) -> dict | None:
        """
        Сюда хорошо смотреть при отладке:
        completion:kwargs содержит tools/tool_choice после того, как Instructor
        превратил response_model в tool schema.  [oai_citation:6‡python.useinstructor.com](https://python.useinstructor.com/concepts/hooks/?utm_source=chatgpt.com)
        """
        if not kwargs:
            return None
        compact = {}
        for k in ("model", "temperature", "max_tokens", "tool_choice"):
            if k in kwargs:
                compact[k] = kwargs[k]
        if "tools" in kwargs and isinstance(kwargs["tools"], list):
            compact["tools"] = f"tools[{len(kwargs['tools'])}]"
        return compact


class AttributeModelFactory:
    """
    В TOOLS режиме нам нужно:
    - явная схема (properties) под каждый атрибут
    - ключи результата должны быть РОВНО атрибутами => используем alias=attr
    - ключи обязаны присутствовать => Field(...), но значение может быть null
    """
    def __init__(self, attributes: list[str]) -> None:
        seen: set[str] = set()
        uniq: list[str] = []
        for a in attributes:
            if a not in seen:
                uniq.append(a)
                seen.add(a)
        self._attributes = uniq

    def build(self):
        fields: dict[str, tuple[object, object]] = {}
        used_names: set[str] = set()

        for attr in self._attributes:
            fname = self._safe_field_name(attr)
            base = fname
            i = 2
            while fname in used_names:
                fname = f"{base}_{i}"
                i += 1
            used_names.add(fname)

            # КЛЮЧ ОБЯЗАН БЫТЬ (Field(...)), значение может быть null (str|None)
            fields[fname] = (str | None, Field(..., alias=attr))

        model_config = ConfigDict(extra="forbid", populate_by_name=True)
        Dynamic = create_model("InsuranceAttributes", __config__=model_config, **fields)  # type: ignore

        # Важно: schema должна быть с alias (русские ключи)
        @classmethod
        def _model_json_schema(cls, *args, **kwargs):
            kwargs.setdefault("by_alias", True)
            return super(Dynamic, cls).model_json_schema(*args, **kwargs)

        Dynamic.model_json_schema = _model_json_schema  # type: ignore
        return Dynamic

    @staticmethod
    def _safe_field_name(s: str) -> str:
        # внутреннее python-имя; внешние ключи НЕ меняются (они alias)
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", s.strip())
        if not s:
            s = "field"
        if re.match(r"^\d", s):
            s = f"f_{s}"
        return s


class SystemPromptBuilder:
    def build(self) -> str:
        return (
            "Ты — сервис извлечения данных из описаний страховых продуктов.\n"
            "Нужно заполнить значения для всех атрибутов из списка.\n"
            "Не выдумывай значения: только из текста. Если не найдено — null.\n"
            "Таблицы markdown часто содержат пары key-value.\n"
            "Верни только структурированный результат (без пояснений).\n"
        )


class InsuranceAttributeExtractionAgent:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._ledger = TokenLedger()
        self._tracer = AgentTracer(config.trace_enabled, config.trace_print, config.trace_max_chars)
        self._system_prompt = SystemPromptBuilder().build()

        # (1) ВКЛЮЧАЕМ tool calling здесь:
        # Instructor будет использовать tools/tool_choice механизм.  [oai_citation:7‡python.useinstructor.com](https://python.useinstructor.com/concepts/patching/?utm_source=chatgpt.com)
        self._client = self._build_instructor_tools_client()

        self._hooks = Hooks()
        self._hooks.on("completion:kwargs", self._ledger.on_completion_kwargs)
        self._hooks.on("completion:response", self._ledger.on_completion_response)

    def extract(
        self,
        markdown_text: str,
        attributes: list[str],
    ) -> tuple[dict[str, str | None] | None, TokenUsageSummary, list[TraceEvent]]:
        response_model = AttributeModelFactory(attributes).build()

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._build_user_payload(markdown_text, attributes)},
        ]

        calls_left = int(self._config.max_total_calls)
        fixes_left = int(self._config.max_format_fixes)

        while calls_left > 0:
            calls_left -= 1

            trace_id = self._tracer.start_call(purpose="tools-extract", messages=messages)

            try:
                # (2) tool schema создаётся ИМЕННО тут: response_model -> tools/tool_choice
                # Это видно в completion:kwargs (tools/tool_choice).  [oai_citation:8‡python.useinstructor.com](https://python.useinstructor.com/concepts/hooks/?utm_source=chatgpt.com)
                obj = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=messages,
                    response_model=response_model,
                    max_retries=0,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_output_tokens,
                    hooks=self._hooks,
                )

                result = obj.model_dump(by_alias=True, exclude_none=False)

                # Доп. проверка: ключи должны совпадать со списком атрибутов
                missing_keys = [a for a in attributes if a not in result]
                extra_keys = [k for k in result.keys() if k not in set(attributes)]
                if missing_keys or extra_keys:
                    raise ValueError(f"missing_keys={missing_keys}, extra_keys={extra_keys}")

                self._tracer.finish_call(
                    trace_id,
                    assistant_text=self._ledger.last_assistant_text(),
                    usage=self._ledger.last_usage(),
                    parsed_result=result,
                    error=None,
                    provider_kwargs=self._ledger.last_provider_kwargs(),
                )
                return result, self._ledger.summary(), self._tracer.events()

            except Exception as e:
                last_text = self._ledger.last_assistant_text()
                self._tracer.finish_call(
                    trace_id,
                    assistant_text=last_text,
                    usage=self._ledger.last_usage(),
                    parsed_result=None,
                    error=str(e),
                    provider_kwargs=self._ledger.last_provider_kwargs(),
                )

                # Любая проблема (включая "нет атрибутов"/не tool-call/битый формат)
                # -> делаем "тонкий" fix БЕЗ документа (экономим токены)
                if fixes_left > 0 and calls_left > 0:
                    fixes_left -= 1
                    messages = self._build_minimal_fix_messages(
                        last_assistant_text=last_text,
                        attributes=attributes,
                        error=str(e),
                    )
                    continue

                break

        return None, self._ledger.summary(), self._tracer.events()

    def _build_instructor_tools_client(self):
        default_headers = {}
        if self._config.litellm_virtual_key:
            default_headers[self._config.litellm_virtual_key_header] = f"Bearer {self._config.litellm_virtual_key}"
            api_key = "not-used"
        else:
            api_key = self._config.litellm_api_key

        base = OpenAI(
            base_url=self._config.litellm_base_url,
            api_key=api_key,
            default_headers=default_headers or None,
        )

        return instructor.patch(base, mode=instructor.Mode.TOOLS)

    @staticmethod
    def _build_user_payload(markdown_text: str, attributes: list[str]) -> str:
        attrs = "\n".join(f"- {a}" for a in attributes)
        return (
            "Извлеки значения атрибутов из текста страхового продукта.\n\n"
            "Список атрибутов (ключи результата должны совпадать с ними ТОЧНО):\n"
            f"{attrs}\n\n"
            "Текст (markdown, включая таблицы):\n"
            "-----\n"
            f"{markdown_text}\n"
            "-----\n"
        )

    def _build_minimal_fix_messages(self, last_assistant_text: str | None, attributes: list[str], error: str) -> list[dict]:
        """
        ВАЖНО: здесь нет исходного markdown.
        Мы просим пересобрать/исправить только результат.
        """
        attrs = "\n".join(f"- {a}" for a in attributes)
        fix_system = (
            "Ты — редактор результата.\n"
            "Твоя задача — вернуть итог строго по списку атрибутов.\n"
            "НЕ добавляй новую информацию и НЕ выдумывай значения.\n"
            "Если значения неизвестны — ставь null.\n"
            "Никаких пояснений.\n"
        )
        fix_user = (
            f"Список атрибутов:\n{attrs}\n\n"
            f"{self._config.format_fix_user_prompt_template.format(error=error)}"
        )

        msgs = [{"role": "system", "content": fix_system}]
        if last_assistant_text:
            msgs.append({"role": "assistant", "content": last_assistant_text})
        msgs.append({"role": "user", "content": fix_user})
        return msgs


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

    # Отладка: посмотреть tools/tool_choice (в provider_kwargs_compact)
    print("\nTRACE EVENTS:", len(trace))
    for ev in trace:
        print(f"\n--- CALL #{ev.index} ---")
        print("provider_kwargs:", ev.provider_kwargs_compact)
        print("usage:", ev.usage)
        print("error:", ev.error)

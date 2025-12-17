# insurance_attr_agent.py

import json
import os
import time

from openai import OpenAI, DefaultHttpxClient
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, RootModel, model_validator


class AgentConfig(BaseModel):
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "CHANGE_ME"

    litellm_virtual_key: str | None = None
    litellm_virtual_key_header: str = "X-Litellm-Key"

    model: str = "gigachat"
    temperature: float = 0.0
    max_output_tokens: int = 1200

    # "JSON" | "TOOLS"
    mode: str = "JSON"

    # Optional: в JSON-режиме можно форсить валидный JSON (если поддерживается)
    # OpenAI Chat Completions: response_format={"type":"json_object"}  [oai_citation:2‡OpenAI Platform](https://platform.openai.com/docs/api-reference/chat?utm_source=chatgpt.com)
    json_force_response_format: bool = False

    max_total_calls: int = 3
    max_fix_calls: int = 1

    connection_retries: int = 1
    connection_retry_sleep_seconds: float = 0.5

    trace_enabled: bool = True
    trace_print: bool = False
    trace_max_chars: int = 6000


class TokenUsageSummary(BaseModel):
    attempts: int = 0
    responses_received: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TraceEvent(BaseModel):
    index: int
    mode: str
    purpose: str
    attempt_no: int

    messages: list[dict[str, str]]

    # TOOLS
    tools_count: int = 0
    tool_choice: dict | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments_raw: str | None = None

    # JSON
    assistant_content: str | None = None

    parsed_result: dict[str, str | None] | None = None
    usage: dict[str, int] | None = None

    error_kind: str | None = None
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

    def start(self, mode: str, purpose: str, attempt_no: int, messages: list[dict], tools: list[dict] | None, tool_choice: dict | None) -> int:
        if not self._enabled:
            return -1
        self._i += 1
        ev = TraceEvent(
            index=self._i,
            mode=mode,
            purpose=purpose,
            attempt_no=attempt_no,
            messages=[self._trim_msg(m) for m in messages],
            tools_count=len(tools or []),
            tool_choice=tool_choice,
        )
        self._events.append(ev)
        if self._do_print:
            print(f"\n[TRACE] CALL #{ev.index} mode={mode} purpose={purpose} attempt={attempt_no}")
        return ev.index

    def finish(
        self,
        trace_id: int,
        *,
        assistant_content: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_arguments_raw: str | None = None,
        parsed_result: dict[str, str | None] | None = None,
        usage: dict[str, int] | None = None,
        error_kind: str | None = None,
        error: str | None = None,
    ) -> None:
        if not self._enabled or trace_id < 0:
            return
        ev = next((x for x in self._events if x.index == trace_id), None)
        if not ev:
            return

        ev.assistant_content = self._trim_text(assistant_content)
        ev.tool_name = tool_name
        ev.tool_call_id = tool_call_id
        ev.tool_arguments_raw = self._trim_text(tool_arguments_raw)
        ev.parsed_result = parsed_result
        ev.usage = usage
        ev.error_kind = error_kind
        ev.error = error

        if self._do_print:
            print(f"[TRACE] RESULT #{trace_id} error_kind={error_kind} usage={usage}")
            if error:
                print(f"[TRACE] error: {error}")

    def _trim_text(self, s: str | None) -> str | None:
        if s is None:
            return None
        if len(s) <= self._max_chars:
            return s
        return s[: self._max_chars] + "…(truncated)"

    def _trim_msg(self, m: dict) -> dict[str, str]:
        return {"role": str(m.get("role", "")), "content": self._trim_text(str(m.get("content", ""))) or ""}


class StrictAttributesResult(RootModel[dict[str, str | None]]):
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
    def __init__(self, tool_name: str = "extract_insurance_attributes") -> None:
        self._tool_name = tool_name

    def build(self, attributes: list[str]) -> tuple[list[dict], dict]:
        props: dict[str, dict] = {}
        for a in attributes:
            props[a] = {"anyOf": [{"type": "string"}, {"type": "null"}]}

        tool = {
            "type": "function",
            "function": {
                "name": self._tool_name,
                "description": "Извлечь значения заданных атрибутов из описания страхового продукта.",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(attributes),
                    "additionalProperties": False,
                },
            },
        }
        tool_choice = {"type": "function", "function": {"name": self._tool_name}}
        return [tool], tool_choice


class StageError(RuntimeError):
    def __init__(self, message: str, last_payload: str) -> None:
        super().__init__(message)
        self.last_payload = last_payload


class InsuranceAttributeExtractionAgent:
    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._mode = (config.mode or "JSON").upper().strip()
        self._client = self._build_openai_client()
        self._tracer = AgentTracer(config.trace_enabled, config.trace_print, config.trace_max_chars)
        self._tool_factory = ToolSchemaFactory()

    def extract(self, markdown_text: str, attributes: list[str]) -> tuple[dict[str, str | None] | None, TokenUsageSummary, list[TraceEvent]]:
        usage_sum = TokenUsageSummary()
        validator_model = StrictAttributesResult.for_attributes(attributes)

        tools = None
        tool_choice = None
        if self._mode == "TOOLS":
            tools, tool_choice = self._tool_factory.build(attributes)

        messages = [
            {"role": "system", "content": self._system_prompt_extract(self._mode)},
            {"role": "user", "content": self._user_payload(markdown_text, attributes, self._mode)},
        ]

        calls_left = int(self._cfg.max_total_calls)
        fix_left = int(self._cfg.max_fix_calls)

        last_payload = ""

        while calls_left > 0:
            calls_left -= 1
            purpose = "extract" if fix_left == self._cfg.max_fix_calls else "fix"

            try:
                result, last_payload = self._call_stage(
                    purpose=purpose,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    validator_model=validator_model,
                    usage_sum=usage_sum,
                )
                return result, usage_sum, self._tracer.events()

            except StageError as e:
                # ВАЖНО: last_payload берём прямо из исключения (он всегда заполнен, если был ответ модели)
                last_payload = e.last_payload

                if fix_left > 0 and calls_left > 0:
                    fix_left -= 1
                    messages = self._minimal_fix_messages(
                        mode=self._mode,
                        attributes=attributes,
                        error=str(e),
                        last_payload=last_payload,
                    )
                    continue

                return None, usage_sum, self._tracer.events()

        return None, usage_sum, self._tracer.events()

    def _call_stage(
        self,
        *,
        purpose: str,
        messages: list[dict],
        tools: list[dict] | None,
        tool_choice: dict | None,
        validator_model: type[StrictAttributesResult],
        usage_sum: TokenUsageSummary,
    ) -> tuple[dict[str, str | None], str]:
        last_payload = ""

        for attempt_no in range(1, int(self._cfg.connection_retries) + 2):
            trace_id = self._tracer.start(self._mode, purpose, attempt_no, messages, tools, tool_choice)
            usage_sum.attempts += 1

            try:
                kwargs = {
                    "model": self._cfg.model,
                    "messages": messages,
                    "temperature": self._cfg.temperature,
                    "max_tokens": self._cfg.max_output_tokens,
                }

                # JSON-mode enforcement (если включено и поддерживается прокси/модель)
                if self._mode == "JSON" and self._cfg.json_force_response_format:
                    kwargs["response_format"] = {"type": "json_object"}  #  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/api-reference/chat?utm_source=chatgpt.com)

                if self._mode == "TOOLS":
                    kwargs["tools"] = tools or []
                    kwargs["tool_choice"] = tool_choice

                resp = self._client.chat.completions.create(**kwargs)

                usage = self._extract_usage(resp)
                if usage:
                    usage_sum.responses_received += 1
                    usage_sum.input_tokens += usage["input_tokens"]
                    usage_sum.output_tokens += usage["output_tokens"]
                    usage_sum.total_tokens += usage["total_tokens"]

                msg = resp.choices[0].message

                if self._mode == "TOOLS":
                    tool_name, tool_call_id, tool_args_raw = self._extract_tool_call_info(msg)
                    if tool_args_raw is None:
                        last_payload = (getattr(msg, "content", None) or "")
                        self._tracer.finish(
                            trace_id,
                            assistant_content=getattr(msg, "content", None),
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            usage=usage,
                            error_kind="protocol",
                            error="TOOLS: нет tool_calls.function.arguments",
                        )
                        raise StageError("TOOLS: нет tool_calls.function.arguments", last_payload)

                    last_payload = tool_args_raw  # <-- сохраняем ДО парсинга

                    try:
                        data = json.loads(tool_args_raw)
                        validated = validator_model.model_validate(data)
                        result = dict(validated.root)
                    except Exception as e:
                        self._tracer.finish(
                            trace_id,
                            assistant_content=getattr(msg, "content", None),
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            tool_arguments_raw=tool_args_raw,
                            usage=usage,
                            error_kind="validation",
                            error=str(e),
                        )
                        raise StageError(f"TOOLS: parse/validate error: {e}", last_payload)

                    self._tracer.finish(
                        trace_id,
                        assistant_content=getattr(msg, "content", None),
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_arguments_raw=tool_args_raw,
                        parsed_result=result,
                        usage=usage,
                        error_kind=None,
                        error=None,
                    )
                    return result, last_payload

                # JSON mode
                assistant_content = getattr(msg, "content", None)
                if not assistant_content:
                    last_payload = ""
                    self._tracer.finish(
                        trace_id,
                        assistant_content=None,
                        usage=usage,
                        error_kind="protocol",
                        error="JSON: пустой message.content",
                    )
                    raise StageError("JSON: пустой message.content", last_payload)

                last_payload = assistant_content  # <-- сохраняем ДО парсинга

                try:
                    data = self._parse_json_object(assistant_content)
                    validated = validator_model.model_validate(data)
                    result = dict(validated.root)
                except Exception as e:
                    self._tracer.finish(
                        trace_id,
                        assistant_content=assistant_content,
                        usage=usage,
                        error_kind="validation",
                        error=str(e),
                    )
                    raise StageError(f"JSON: parse/validate error: {e}", last_payload)

                self._tracer.finish(
                    trace_id,
                    assistant_content=assistant_content,
                    parsed_result=result,
                    usage=usage,
                    error_kind=None,
                    error=None,
                )
                return result, last_payload

            except (APIConnectionError, APITimeoutError) as e:
                self._tracer.finish(
                    trace_id,
                    error_kind="connection",
                    error=str(e),
                )
                if attempt_no <= int(self._cfg.connection_retries):
                    time.sleep(float(self._cfg.connection_retry_sleep_seconds))
                    continue
                raise StageError(f"Connection error: {e}", last_payload)

        raise StageError("Unexpected retry loop exit", last_payload)

    def _build_openai_client(self) -> OpenAI:
        headers = {}
        if self._cfg.litellm_virtual_key:
            headers[self._cfg.litellm_virtual_key_header] = f"Bearer {self._cfg.litellm_virtual_key}"
            api_key = "not-used"
        else:
            api_key = self._cfg.litellm_api_key

        http_client = DefaultHttpxClient(verify=False)
        return OpenAI(
            base_url=self._cfg.litellm_base_url,
            api_key=api_key,
            default_headers=headers or None,
            http_client=http_client,
            max_retries=0,
        )

    @staticmethod
    def _system_prompt_extract(mode: str) -> str:
        if mode == "TOOLS":
            return (
                "Ты извлекаешь значения атрибутов из текста страхового продукта.\n"
                "Всегда вызывай инструмент и возвращай аргументы инструмента.\n"
                "Значения бери только из входного текста. Если не найдено — null.\n"
                "Никаких пояснений.\n"
            )
        return (
            "Ты извлекаешь значения атрибутов из текста страхового продукта.\n"
            "Верни ТОЛЬКО валидный JSON-объект.\n"
            "Ключи всегда в ДВОЙНЫХ кавычках.\n"
            "Значения — строки или null.\n"
            "Никаких пояснений, никакого markdown, никаких ```.\n"
        )

    @staticmethod
    def _user_payload(markdown_text: str, attributes: list[str], mode: str) -> str:
        attrs_json = json.dumps(attributes, ensure_ascii=False)
        return (
            "attributes_json = " + attrs_json + "\n\n"
            "Текст (markdown, включая таблицы):\n"
            "-----\n"
            f"{markdown_text}\n"
            "-----\n"
            + ("\nВсегда вызови инструмент.\n" if mode == "TOOLS" else "")
        )

    @staticmethod
    def _minimal_fix_messages(mode: str, attributes: list[str], error: str, last_payload: str) -> list[dict]:
        # Ключевое: передаём атрибуты как JSON array и даём валидный JSON template.
        attrs_json = json.dumps(attributes, ensure_ascii=False)
        template_json = json.dumps({a: None for a in attributes}, ensure_ascii=False, indent=2)

        if mode == "TOOLS":
            sys = (
                "Ты исправляешь РЕЗУЛЬТАТ.\n"
                "Всегда вызывай инструмент и возвращай аргументы инструмента.\n"
                "НЕ выдумывай новые значения, только исправь структуру.\n"
                "Ключи должны быть строго как в attributes_json.\n"
                "Никаких пояснений.\n"
            )
        else:
            sys = (
                "Ты исправляешь JSON-ответ.\n"
                "Верни ТОЛЬКО валидный JSON-объект.\n"
                "Ключи всегда в ДВОЙНЫХ кавычках.\n"
                "Ключи строго как в attributes_json.\n"
                "Лишние ключи запрещены.\n"
                "Никаких ``` и markdown.\n"
            )

        usr = (
            f"Ошибка: {error}\n\n"
            f"attributes_json = {attrs_json}\n\n"
            "template_json (пример структуры, ключи уже в кавычках):\n"
            f"{template_json}\n\n"
            "Вот предыдущий ответ модели (исправь ЕГО):\n"
            "-----\n"
            f"{last_payload if last_payload else '[EMPTY_LAST_PAYLOAD]'}\n"
            "-----\n"
            "Важно: это должен быть ВАЛИДНЫЙ JSON (ключи в двойных кавычках).\n"
        )

        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    @staticmethod
    def _extract_tool_call_info(message) -> tuple[str | None, str | None, str | None]:
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None, None, None
        tc0 = tool_calls[0]
        tool_call_id = getattr(tc0, "id", None)
        fn = getattr(tc0, "function", None)
        if not fn:
            return None, str(tool_call_id) if tool_call_id is not None else None, None
        tool_name = getattr(fn, "name", None)
        args = getattr(fn, "arguments", None)
        return (
            str(tool_name) if tool_name is not None else None,
            str(tool_call_id) if tool_call_id is not None else None,
            str(args) if args is not None else None,
        )

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
    def _parse_json_object(text: str) -> dict:
        t = text.strip()

        # если модель завернула в ```json ... ```
        if t.startswith("```"):
            t = t.strip("`").strip()
            if t.lower().startswith("json"):
                t = t[4:].strip()

        return json.loads(t)


# --- Example usage ---
if __name__ == "__main__":
    md_text = "# ... ваш markdown ..."
    attrs = ["Страховая сумма", "Франшиза"]

    # JSON mode
    cfg_json = AgentConfig(
        litellm_base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"),
        litellm_api_key=os.getenv("LITELLM_API_KEY", "CHANGE_ME"),
        model=os.getenv("LLM_MODEL", "gigachat"),
        mode="JSON",
        json_force_response_format=False,  # включи True если прокси/модель поддерживает  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/api-reference/chat?utm_source=chatgpt.com)
        trace_print=True,
    )
    agent_json = InsuranceAttributeExtractionAgent(cfg_json)
    r, u, t = agent_json.extract(md_text, attrs)

    # TOOLS mode
    cfg_tools = cfg_json.model_copy(update={"mode": "TOOLS"})
    agent_tools = InsuranceAttributeExtractionAgent(cfg_tools)
    r2, u2, t2 = agent_tools.extract(md_text, attrs)

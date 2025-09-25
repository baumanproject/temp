# -*- coding: utf-8 -*-
import os, re, json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
 
class SummaryOutput(BaseModel):
    """
    Итоговая выжимка для аналитика по компании+периоду (на основе набора прошедших фильтр новостей).
    """
    stance: str = Field(..., description="Тональность/оценка: 'Позитивно' | 'Нейтрально' | 'Негативно'.")
    thesis: str = Field(..., description="Главный тезис (1–2 предложения).")
    key_facts: List[str] = Field(..., description="Топ-5 фактов/метрик/наблюдений из контекста (операционные).")
    drivers_up: List[str] = Field(..., description="Положительные драйверы/катализаторы.")
    drivers_down: List[str] = Field(..., description="Сдерживающие факторы/риски.")
    forecast_points: List[str] = Field(..., description="Тезисы о прогнозах/рекомендациях (если встречались).")
    recommendation: str = Field(..., description="Рекомендация: 'Покупать' | 'Держать' | 'Продавать' | 'Нет рекомендации'.")
    rationale: str = Field(..., description="Короткое пояснение рекомендации.")
    evidence_spans: List[str] = Field(..., description="До 5 цитат из входа в подтверждение ключевых пунктов.")
    confidence: str = Field(..., description="Доверие: 'низкое' | 'среднее' | 'высокое'.")
 
    model_config = {"extra": "forbid"}
 
class _SummPromptCapture(BaseCallbackHandler):
    def __init__(self, sink): self.sink = sink
    def on_chat_model_start(self, serialized, messages, **kwargs):
        batch = messages[0] if messages else []
        norm = []
        for m in batch:
            role = getattr(m, "type", None) or m.__class__.__name__.replace("Message", "").lower()
            norm.append({"role": role, "content": getattr(m, "content", str(m))})
        self.sink.last_prompt_messages = norm
        self.sink.last_prompt_text = "\n".join(f"{x['role'].upper()}: {x['content']}" for x in norm)
 
class OperationalSummaryLC:
    """
    Одним вызовом LLM делает выжимку по N прошедшим фильтр новостям (компания+период).
    Параметры: reasoning, llm_max_retries, chain_retry_attempts; колбэки для логирования промпта.
    """
 
    def __init__(
        self,
        base_url: str = "http://localhost",
        base_port: int = 4000,
        model_name: str = "qwen3-a30-3b",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        structured_method: str = "function_calling",  # 'json_schema' | 'function_calling' | 'json_mode'
        strict: Optional[bool] = None,
        include_raw: bool = False,
        reasoning: bool = True,
        llm_max_retries: int = 2,
        chain_retry_attempts: int = 1,
        log_prompts: bool = True
    ):
        model_kwargs = {"enable_thinking": True, "thinking": {"budget": 2048}, "reasoning": {"effort": "medium"}} if reasoning else None
 
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-local"),
            base_url=f"{base_url}:{base_port}/v1",
            max_retries=llm_max_retries,
            model_kwargs=model_kwargs
        )
 
        self.structured = self.llm.with_structured_output(
            SummaryOutput, method=structured_method, strict=strict, include_raw=include_raw
        ).with_retry(stop_after_attempt=max(1, chain_retry_attempts))
 
        self.log_prompts = bool(log_prompts)
        self.last_prompt_text: Optional[str] = None
        self.last_prompt_messages: Optional[List[Dict[str, str]]] = None
        self._cb = _SummPromptCapture(self)
 
    # ----- подготовка контекста -----
 
    def _clip(self, s: str, max_len: int) -> str:
        return s if len(s) <= max_len else s[:max_len] + "…"
 
    def _build_context(self, items: List[str], max_items: int, max_chars: int) -> str:
        lines = []
        per_item = max(300, max_chars // max(1, max_items))  # простая эвристика
        for i, t in enumerate(items[:max_items], 1):
            lines.append(f"{i}) " + self._clip(t.replace("\n", " "), per_item))
        return "\n".join(lines)
 
    def _make_system_prompt(self, company: str, period: str) -> str:
        return (
            "Ты — инвестиционный аналитик. На основе перечня релевантных новостей по ОПЕРАЦИОННОЙ отчётности "
            f"сделай краткую выжимку для компании '{company}' за период '{period}'. "
            "Строго следуй схеме SummaryOutput и выведи только JSON.\n"
            "Фокус: ключевые факты (операционные показатели), динамика, драйверы/риски, встречавшиеся прогнозы, итоговая рекомендация."
        )
 
    # ----- вызов -----
 
    def summarize(self, company: str, period: str, texts: List[str], max_items: int = 8, max_context_chars: int = 8000) -> SummaryOutput:
        context = self._build_context(texts, max_items=max_items, max_chars=max_context_chars)
        sys_msg = self._make_system_prompt(company, period)
        messages = [("system", sys_msg), ("user", "Релевантные фрагменты:\n" + context)]
        out = self.structured.invoke(messages, config={"callbacks": [self._cb]} if self.log_prompts else None)
        return out["parsed"] if isinstance(out, dict) and "parsed" in out else out
 
    def get_last_prompt_text(self) -> Optional[str]:
        return self.last_prompt_text

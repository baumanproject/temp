import ast
import asyncio
import os
import sys
from decimal import Decimal

from langchain.agents import create_agent
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, SecretStr


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "Арифметическое выражение. Разрешены только числа, пробелы, "
            "скобки и операторы +, -, *, /."
        ),
        examples=["15 * (7 + 3)"],
    )


class CalculatorOutput(BaseModel):
    status: str = Field(description="Статус выполнения: success или error.")
    value: str | None = Field(
        default=None,
        description="Результат вычисления в виде строки.",
    )
    error: str | None = Field(
        default=None,
        description="Описание ошибки, если вычисление не выполнено.",
    )


class SafeExpressionCalculator:
    def calculate(self, expression: str) -> str:
        try:
            parsed_expression = ast.parse(expression, mode="eval")
            value = self._evaluate_node(parsed_expression.body)

            return CalculatorOutput(
                status="success",
                value=str(value),
                error=None,
            ).model_dump_json(ensure_ascii=False)

        except Exception as error:
            return CalculatorOutput(
                status="error",
                value=None,
                error=str(error),
            ).model_dump_json(ensure_ascii=False)

    def _evaluate_node(self, node) -> Decimal:
        if isinstance(node, ast.Constant):
            return self._evaluate_constant(node)

        if isinstance(node, ast.BinOp):
            return self._evaluate_binary_operation(node)

        if isinstance(node, ast.UnaryOp):
            return self._evaluate_unary_operation(node)

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    def _evaluate_constant(self, node: ast.Constant) -> Decimal:
        if not isinstance(node.value, int | float):
            raise ValueError("Only int and float constants are allowed")

        return Decimal(str(node.value))

    def _evaluate_binary_operation(self, node: ast.BinOp) -> Decimal:
        left_value = self._evaluate_node(node.left)
        right_value = self._evaluate_node(node.right)

        if isinstance(node.op, ast.Add):
            return left_value + right_value

        if isinstance(node.op, ast.Sub):
            return left_value - right_value

        if isinstance(node.op, ast.Mult):
            return left_value * right_value

        if isinstance(node.op, ast.Div):
            if right_value == 0:
                raise ValueError("Division by zero")

            return left_value / right_value

        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    def _evaluate_unary_operation(self, node: ast.UnaryOp) -> Decimal:
        value = self._evaluate_node(node.operand)

        if isinstance(node.op, ast.UAdd):
            return value

        if isinstance(node.op, ast.USub):
            return -value

        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")


class CalculatorToolFactory:
    def build(self) -> StructuredTool:
        calculator = SafeExpressionCalculator()

        return StructuredTool.from_function(
            func=calculator.calculate,
            name="calculator",
            description=(
                "Выполняет безопасные арифметические вычисления. "
                "Используй этот tool только когда пользователю нужен точный расчет. "
                "Аргумент expression должен содержать только числа, скобки, пробелы "
                "и операторы +, -, *, /. "
                "Возвращает JSON-строку формата "
                '{"status":"success|error","value":"string|null","error":"string|null"}.'
            ),
            args_schema=CalculatorInput,
        )


class GigaChatRuntimeConfig(BaseModel):
    model: str = Field(default="GigaChat-2-Pro")
    base_url: str | None = Field(default=None)
    auth_url: str | None = Field(default=None)
    access_token: SecretStr | None = Field(default=None)
    credentials: SecretStr | None = Field(default=None)
    scope: str | None = Field(default=None)
    verify_ssl_certs: bool = Field(default=False)

    temperature: float = Field(default=0.0)
    top_p: float = Field(default=0.8)
    max_tokens: int = Field(default=1024)
    timeout: float = Field(default=60.0)
    max_retries: int = Field(default=2)


class GigaChatRuntimeConfigFactory:
    def build_from_environment(self) -> GigaChatRuntimeConfig:
        access_token = os.getenv("GIGACHAT_ACCESS_TOKEN")
        credentials = os.getenv("GIGACHAT_CREDENTIALS")

        if access_token is None and credentials is None:
            raise ValueError(
                "Set either GIGACHAT_ACCESS_TOKEN or GIGACHAT_CREDENTIALS"
            )

        return GigaChatRuntimeConfig(
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro"),
            base_url=os.getenv("GIGACHAT_BASE_URL"),
            auth_url=os.getenv("GIGACHAT_AUTH_URL"),
            access_token=self._build_secret(access_token),
            credentials=self._build_secret(credentials),
            scope=os.getenv("GIGACHAT_SCOPE"),
            verify_ssl_certs=self._read_bool(
                name="GIGACHAT_VERIFY_SSL",
                default=False,
            ),
            temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("GIGACHAT_TOP_P", "0.8")),
            max_tokens=int(os.getenv("GIGACHAT_MAX_TOKENS", "1024")),
            timeout=float(os.getenv("GIGACHAT_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("GIGACHAT_MAX_RETRIES", "2")),
        )

    def _build_secret(self, value: str | None) -> SecretStr | None:
        if value is None:
            return None

        return SecretStr(value)

    def _read_bool(self, name: str, default: bool) -> bool:
        raw_value = os.getenv(name)

        if raw_value is None:
            return default

        return raw_value.lower() in {"1", "true", "yes", "y"}


class GigaChatModelFactory:
    def build(self, config: GigaChatRuntimeConfig) -> GigaChat:
        return GigaChat(
            model=config.model,
            base_url=config.base_url,
            auth_url=config.auth_url,
            access_token=self._read_secret(config.access_token),
            credentials=self._read_secret(config.credentials),
            scope=config.scope,
            verify_ssl_certs=config.verify_ssl_certs,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            function_ranker={"enabled": False},
            allow_any_tool_choice_fallback=True,
        )

    def _read_secret(self, value: SecretStr | None) -> str | None:
        if value is None:
            return None

        return value.get_secret_value()


class GigaChatCalculatorAgentFactory:
    def build(self):
        config = GigaChatRuntimeConfigFactory().build_from_environment()
        model = GigaChatModelFactory().build(config)
        calculator_tool = CalculatorToolFactory().build()

        return create_agent(
            model=model,
            tools=[calculator_tool],
            system_prompt=(
                "Ты русскоязычный помощник. "
                "Если пользователь просит посчитать выражение, используй tool calculator. "
                "Не считай арифметику самостоятельно, если можно вызвать calculator. "
                "После получения результата tool дай короткий финальный ответ на русском языке."
            ),
        )


class GigaChatCalculatorClient:
    def __init__(self) -> None:
        self._agent = GigaChatCalculatorAgentFactory().build()

    async def ask_async(self, user_text: str) -> str:
        result = await self._agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=user_text),
                ]
            },
            config={
                "recursion_limit": 5,
            },
        )

        return self._extract_final_answer(result)

    def ask(self, user_text: str) -> str:
        return asyncio.run(self.ask_async(user_text))

    def _extract_final_answer(self, result: dict) -> str:
        messages = result["messages"]

        if not messages:
            raise ValueError("Agent returned empty messages")

        final_message = messages[-1]
        content = final_message.content

        if isinstance(content, str):
            return content

        return str(content)


def ask_gigachat_calculator(user_text: str) -> str:
    return GigaChatCalculatorClient().ask(user_text)


def main() -> None:
    user_text = " ".join(sys.argv[1:]).strip()

    if not user_text:
        user_text = "Посчитай 15 * (7 + 3)."

    answer = ask_gigachat_calculator(user_text)

    print(answer)


if __name__ == "__main__":
    main()

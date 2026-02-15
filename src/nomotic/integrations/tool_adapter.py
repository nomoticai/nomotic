"""LLM Tool-Use Adapter — bridges LLM tool calls to governed execution.

Maps tool call requests from Claude, OpenAI, or any LLM that uses
the tool-use pattern into GovernedToolExecutor evaluations.

Usage with Claude:
    from nomotic import GovernedToolExecutor
    from nomotic.integrations.tool_adapter import ToolAdapter

    executor = GovernedToolExecutor.connect("claims-bot")
    adapter = ToolAdapter(executor, tools=my_tools)

    # In your agent loop:
    for tool_call in response.content:
        if tool_call.type == "tool_use":
            result = adapter.handle_tool_call(
                tool_name=tool_call.name,
                tool_args=tool_call.input,
            )
            # Send result back to Claude

Usage with OpenAI:
    for tool_call in response.choices[0].message.tool_calls:
        result = adapter.handle_tool_call(
            tool_name=tool_call.function.name,
            tool_args=json.loads(tool_call.function.arguments),
        )
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from nomotic.executor import ExecutionResult, GovernedToolExecutor

__all__ = [
    "GovernanceDeniedError",
    "ToolAdapter",
    "ToolDefinition",
    "ToolRegistry",
    "govern_tool_call",
]


@dataclass
class ToolDefinition:
    """A registered tool with its function and governance metadata."""

    name: str
    function: Callable
    description: str = ""
    target: str = ""  # default governance target (can be overridden per call)
    action_type: str = ""  # default governance action type (defaults to tool name)
    extract_target: Callable[[dict], str] | None = None  # extract target from args


class ToolRegistry:
    """Registry of tools available for governed execution.

    Usage:
        tools = ToolRegistry()

        @tools.register(description="Query a database table", target="database")
        def query_db(table: str, sql: str) -> list[dict]:
            return db.execute(sql)

        @tools.register(description="Read a file")
        def read_file(path: str) -> str:
            return open(path).read()

        # Or register without decorator:
        tools.add("send_email", send_email_fn, description="Send an email")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        description: str = "",
        target: str = "",
        action_type: str = "",
        extract_target: Callable[[dict], str] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to register a function as a governed tool."""

        def decorator(fn: Callable) -> Callable:
            self._tools[fn.__name__] = ToolDefinition(
                name=fn.__name__,
                function=fn,
                description=description,
                target=target,
                action_type=action_type or fn.__name__,
                extract_target=extract_target,
            )
            return fn

        return decorator

    def add(
        self,
        name: str,
        function: Callable,
        description: str = "",
        target: str = "",
        action_type: str = "",
        extract_target: Callable[[dict], str] | None = None,
    ) -> None:
        """Register a tool without using the decorator pattern."""
        self._tools[name] = ToolDefinition(
            name=name,
            function=function,
            description=description,
            target=target,
            action_type=action_type or name,
            extract_target=extract_target,
        )

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_claude_tools(self) -> list[dict]:
        """Export tool definitions in Claude API format.

        Returns a list of tool definitions compatible with the
        Anthropic messages API tools parameter.

        Note: This generates basic schemas. For production use,
        developers should define their own detailed input_schema.
        """
        tools = []
        for td in self._tools.values():
            tool = {
                "name": td.name,
                "description": td.description or td.name,
                "input_schema": self._infer_schema(td.function),
            }
            tools.append(tool)
        return tools

    def to_openai_tools(self) -> list[dict]:
        """Export tool definitions in OpenAI API format."""
        tools = []
        for td in self._tools.values():
            tool = {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description or td.name,
                    "parameters": self._infer_schema(td.function),
                },
            }
            tools.append(tool)
        return tools

    @staticmethod
    def _infer_schema(fn: Callable) -> dict:
        """Infer a basic JSON schema from function type hints."""
        sig = inspect.signature(fn)
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            prop: dict[str, str] = {}
            if param.annotation == str:
                prop["type"] = "string"
            elif param.annotation == int:
                prop["type"] = "integer"
            elif param.annotation == float:
                prop["type"] = "number"
            elif param.annotation == bool:
                prop["type"] = "boolean"
            else:
                prop["type"] = "string"  # fallback
            properties[param_name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


class ToolAdapter:
    """Bridges LLM tool calls to governed execution.

    Takes tool call requests from any LLM (Claude, OpenAI, etc.)
    and routes them through the GovernedToolExecutor.
    """

    def __init__(
        self,
        executor: GovernedToolExecutor,
        tools: ToolRegistry,
        *,
        on_denied: str = "message",  # "message", "raise", "silent"
        denied_message_template: str = "Action denied by governance: {reason}",
    ) -> None:
        """
        Args:
            executor: GovernedToolExecutor connected to an agent identity
            tools: Registry of available tools
            on_denied: How to handle denials:
                "message" — return a message explaining the denial (default)
                "raise" — raise GovernanceDeniedError
                "silent" — return None
            denied_message_template: Template for denial messages
                ({reason}, {action}, {target} available)
        """
        self._executor = executor
        self._tools = tools
        self._on_denied = on_denied
        self._denied_template = denied_message_template

    def handle_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Any:
        """Handle a single tool call from an LLM.

        1. Looks up the tool in the registry
        2. Determines governance action and target
        3. Evaluates through GovernedToolExecutor
        4. Executes if allowed, returns denial if not

        Args:
            tool_name: The tool name from the LLM's tool call
            tool_args: The arguments from the LLM's tool call

        Returns:
            Tool output if allowed, denial message/None/exception if denied
        """
        tool_def = self._tools.get(tool_name)
        if tool_def is None:
            return f"Unknown tool: {tool_name}"

        # Determine governance target
        target = tool_def.target
        if tool_def.extract_target:
            target = tool_def.extract_target(tool_args)
        elif not target:
            # Try common argument names for target
            for key in (
                "target",
                "table",
                "path",
                "file",
                "resource",
                "database",
                "recipient",
            ):
                if key in tool_args:
                    target = str(tool_args[key])
                    break

        # Determine governance action type
        action_type = tool_def.action_type or tool_name

        # Execute through governance
        result = self._executor.execute(
            action=action_type,
            target=target,
            params=tool_args,
            tool_fn=lambda: tool_def.function(**tool_args),
        )

        if result.allowed:
            return result.data
        else:
            return self._handle_denial(result, action_type, target)

    def handle_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Handle multiple tool calls (batch from a single LLM response).

        Args:
            tool_calls: List of {"name": str, "args": dict} or
                       {"name": str, "input": dict} (Claude format)

        Returns:
            List of {"tool_name": str, "result": Any, "allowed": bool}
        """
        results = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or tc.get("input") or tc.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)

            output = self.handle_tool_call(name, args)
            results.append({
                "tool_name": name,
                "result": output,
                "allowed": not isinstance(output, str)
                or "denied" not in output.lower(),
            })
        return results

    def _handle_denial(
        self, result: ExecutionResult, action: str, target: str
    ) -> Any:
        """Handle a governance denial based on configured behavior."""
        if self._on_denied == "raise":
            raise GovernanceDeniedError(
                f"Governance denied {action} on {target}: {result.reason}",
                result=result,
            )
        elif self._on_denied == "silent":
            return None
        else:  # "message"
            return self._denied_template.format(
                reason=result.reason,
                action=action,
                target=target,
                verdict=result.verdict,
                ucs=result.ucs,
            )


class GovernanceDeniedError(Exception):
    """Raised when governance denies a tool call (when on_denied="raise")."""

    def __init__(self, message: str, result: ExecutionResult) -> None:
        super().__init__(message)
        self.result = result


def govern_tool_call(
    executor: GovernedToolExecutor,
    tool_name: str,
    tool_args: dict[str, Any],
    tools: ToolRegistry,
    **kwargs: Any,
) -> Any:
    """One-shot convenience function for governing a single tool call.

    Usage:
        result = govern_tool_call(executor, "query_db", {"table": "customers"}, tools)
    """
    adapter = ToolAdapter(executor, tools, **kwargs)
    return adapter.handle_tool_call(tool_name, tool_args)

"""Tests for LLM Tool-Use Adapter — ToolRegistry and ToolAdapter."""

import pytest

from nomotic.authority import CertificateAuthority
from nomotic.executor import GovernedToolExecutor
from nomotic.integrations.tool_adapter import (
    GovernanceDeniedError,
    ToolAdapter,
    ToolRegistry,
    govern_tool_call,
)
from nomotic.keys import SigningKey
from nomotic.sandbox import AgentConfig, save_agent_config
from nomotic.store import FileCertificateStore


def _setup_agent(
    tmp_path,
    agent_id: str = "TestBot",
    actions: list[str] | None = None,
    boundaries: list[str] | None = None,
) -> str:
    """Create a certificate and config for an agent in tmp_path.

    Returns the certificate ID.
    """
    sk, _vk = SigningKey.generate()
    store = FileCertificateStore(tmp_path)
    ca = CertificateAuthority(issuer_id="test-issuer", signing_key=sk, store=store)

    cert, _agent_sk = ca.issue(
        agent_id=agent_id,
        archetype="general",
        organization="test-org",
        zone_path="global",
        owner="test-owner",
    )

    if actions is None:
        actions = ["read", "write", "query"]
    config = AgentConfig(
        agent_id=agent_id,
        actions=actions,
        boundaries=boundaries or [],
    )
    save_agent_config(tmp_path, config)

    return cert.certificate_id


class TestToolRegistry:
    def test_register_decorator(self):
        tools = ToolRegistry()

        @tools.register(description="Read a file")
        def read_file(path: str) -> str:
            return f"contents of {path}"

        assert "read_file" in tools.names()
        assert tools.get("read_file").description == "Read a file"

    def test_add_method(self):
        tools = ToolRegistry()
        tools.add("query", lambda table: f"queried {table}", description="Query DB")
        assert "query" in tools.names()
        assert tools.get("query").description == "Query DB"

    def test_get_unknown_returns_none(self):
        tools = ToolRegistry()
        assert tools.get("nonexistent") is None

    def test_names_returns_all_registered(self):
        tools = ToolRegistry()
        tools.add("a", lambda: None)
        tools.add("b", lambda: None)
        tools.add("c", lambda: None)
        assert tools.names() == ["a", "b", "c"]

    def test_register_preserves_function(self):
        tools = ToolRegistry()

        @tools.register(description="Test")
        def my_func(x: int) -> int:
            return x * 2

        # Decorator should return the original function
        assert my_func(5) == 10

    def test_default_action_type_is_function_name(self):
        tools = ToolRegistry()

        @tools.register(description="Read a file")
        def read_file(path: str) -> str:
            return ""

        assert tools.get("read_file").action_type == "read_file"

    def test_custom_action_type(self):
        tools = ToolRegistry()
        tools.add("read", lambda: None, action_type="file_read")
        assert tools.get("read").action_type == "file_read"

    def test_extract_target_stored(self):
        extractor = lambda args: args.get("table", "")
        tools = ToolRegistry()
        tools.add("query", lambda table: None, extract_target=extractor)
        assert tools.get("query").extract_target is extractor

    def test_to_claude_tools(self):
        tools = ToolRegistry()

        @tools.register(description="Read a file")
        def read_file(path: str) -> str:
            return ""

        claude_tools = tools.to_claude_tools()
        assert len(claude_tools) == 1
        assert claude_tools[0]["name"] == "read_file"
        assert claude_tools[0]["description"] == "Read a file"
        assert "input_schema" in claude_tools[0]
        schema = claude_tools[0]["input_schema"]
        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert "path" in schema["required"]

    def test_to_openai_tools(self):
        tools = ToolRegistry()

        @tools.register(description="Send an email")
        def send_email(recipient: str, body: str) -> bool:
            return True

        openai_tools = tools.to_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        func = openai_tools[0]["function"]
        assert func["name"] == "send_email"
        assert func["description"] == "Send an email"
        assert "recipient" in func["parameters"]["properties"]
        assert "body" in func["parameters"]["properties"]

    def test_infer_schema_type_hints(self):
        tools = ToolRegistry()

        @tools.register(description="Multi-type function")
        def multi(name: str, count: int, ratio: float, flag: bool) -> None:
            pass

        schema = tools.to_claude_tools()[0]["input_schema"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"
        assert schema["properties"]["flag"]["type"] == "boolean"
        assert set(schema["required"]) == {"name", "count", "ratio", "flag"}

    def test_infer_schema_optional_params(self):
        tools = ToolRegistry()

        @tools.register(description="With defaults")
        def func(required_arg: str, optional_arg: str = "default") -> None:
            pass

        schema = tools.to_claude_tools()[0]["input_schema"]
        assert "required_arg" in schema["required"]
        assert "optional_arg" not in schema["required"]

    def test_infer_schema_no_annotation(self):
        tools = ToolRegistry()
        tools.add("untyped", lambda x: x)
        schema = tools.to_claude_tools()[0]["input_schema"]
        # Unannotated params default to "string"
        assert schema["properties"]["x"]["type"] == "string"

    def test_to_claude_tools_multiple(self):
        tools = ToolRegistry()
        tools.add("a", lambda: None, description="Tool A")
        tools.add("b", lambda: None, description="Tool B")
        claude_tools = tools.to_claude_tools()
        assert len(claude_tools) == 2
        names = {t["name"] for t in claude_tools}
        assert names == {"a", "b"}

    def test_description_fallback_to_name(self):
        tools = ToolRegistry()
        tools.add("my_tool", lambda: None)  # no description
        claude_tools = tools.to_claude_tools()
        assert claude_tools[0]["description"] == "my_tool"


class TestToolAdapter:
    def test_allowed_tool_executes(self, tmp_path):
        """Tool function runs when governance approves."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"contents of {path}")

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        result = adapter.handle_tool_call("read_file", {"path": "test.txt"})
        assert result == "contents of test.txt"

    def test_denied_tool_returns_message(self, tmp_path):
        """Denied tool returns a denial message by default."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        tools = ToolRegistry()
        tools.add("delete_file", lambda path: None)

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        result = adapter.handle_tool_call("delete_file", {"path": "important.csv"})
        assert "denied" in result.lower()

    def test_denied_tool_raises(self, tmp_path):
        """Denied tool raises GovernanceDeniedError when on_denied='raise'."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        tools = ToolRegistry()
        tools.add("delete_file", lambda path: None)

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools, on_denied="raise")

        with pytest.raises(GovernanceDeniedError) as exc_info:
            adapter.handle_tool_call("delete_file", {"path": "x"})
        assert exc_info.value.result is not None
        assert exc_info.value.result.allowed is False

    def test_denied_tool_silent(self, tmp_path):
        """Denied tool returns None when on_denied='silent'."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        tools = ToolRegistry()
        tools.add("delete_file", lambda path: None)

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools, on_denied="silent")

        result = adapter.handle_tool_call("delete_file", {"path": "x"})
        assert result is None

    def test_unknown_tool(self, tmp_path):
        """Unknown tool returns error message."""
        _setup_agent(tmp_path, "TestBot")
        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, ToolRegistry())

        result = adapter.handle_tool_call("nonexistent", {})
        assert "Unknown tool" in result

    def test_extract_target_from_args(self, tmp_path):
        """extract_target callback extracts governance target from tool args."""
        _setup_agent(tmp_path, "TestBot", actions=["query_db"])
        tools = ToolRegistry()
        tools.add(
            "query_db",
            lambda table, sql: f"results from {table}",
            extract_target=lambda args: args.get("table", ""),
        )

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        result = adapter.handle_tool_call(
            "query_db", {"table": "customers", "sql": "SELECT *"}
        )
        assert result == "results from customers"

    def test_auto_target_from_common_args(self, tmp_path):
        """Target is automatically extracted from common arg names."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"read {path}")

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        # "path" is in the common argument names list
        result = adapter.handle_tool_call("read_file", {"path": "data.csv"})
        assert result == "read data.csv"

    def test_static_target(self, tmp_path):
        """Static target from ToolDefinition is used when set."""
        _setup_agent(tmp_path, "TestBot", actions=["list_tables"])
        tools = ToolRegistry()
        tools.add(
            "list_tables", lambda: ["users", "orders"], target="database"
        )

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        result = adapter.handle_tool_call("list_tables", {})
        assert result == ["users", "orders"]

    def test_batch_tool_calls(self, tmp_path):
        """handle_tool_calls processes multiple calls and returns results."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"contents of {path}")
        tools.add("delete_file", lambda path: None)  # will be denied

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        results = adapter.handle_tool_calls([
            {"name": "read_file", "args": {"path": "a.txt"}},
            {"name": "delete_file", "args": {"path": "b.txt"}},
        ])
        assert len(results) == 2
        assert results[0]["tool_name"] == "read_file"
        assert results[0]["result"] == "contents of a.txt"
        assert results[0]["allowed"] is True
        assert results[1]["tool_name"] == "delete_file"
        assert results[1]["allowed"] is False

    def test_batch_with_claude_format(self, tmp_path):
        """handle_tool_calls accepts Claude 'input' format."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"contents of {path}")

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        results = adapter.handle_tool_calls([
            {"name": "read_file", "input": {"path": "test.txt"}},
        ])
        assert results[0]["result"] == "contents of test.txt"

    def test_batch_with_string_arguments(self, tmp_path):
        """handle_tool_calls parses JSON string arguments (OpenAI format)."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"contents of {path}")

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(executor, tools)

        results = adapter.handle_tool_calls([
            {"name": "read_file", "arguments": '{"path": "test.txt"}'},
        ])
        assert results[0]["result"] == "contents of test.txt"

    def test_custom_denied_message_template(self, tmp_path):
        """Custom denied message template uses format placeholders."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        tools = ToolRegistry()
        tools.add("delete_file", lambda path: None)

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)
        adapter = ToolAdapter(
            executor,
            tools,
            denied_message_template="BLOCKED: {action} on {target} - {reason}",
        )

        result = adapter.handle_tool_call("delete_file", {"path": "x"})
        assert "BLOCKED" in result
        assert "delete_file" in result


class TestGovernanceDeniedError:
    def test_has_result(self):
        """GovernanceDeniedError stores the ExecutionResult."""
        from nomotic.executor import ExecutionResult

        result = ExecutionResult(
            allowed=False,
            verdict="DENY",
            reason="out of scope",
            data=None,
            ucs=0.3,
            tier=1,
            trust_before=0.5,
            trust_after=0.45,
            trust_delta=-0.05,
            dimension_scores={},
            vetoed_by=["scope"],
            action_id="test-123",
            duration_ms=1.0,
        )
        err = GovernanceDeniedError("test denied", result=result)
        assert str(err) == "test denied"
        assert err.result is result
        assert err.result.verdict == "DENY"


class TestGovernToolCall:
    def test_convenience_function(self, tmp_path):
        """govern_tool_call is a one-shot convenience wrapper."""
        _setup_agent(tmp_path, "TestBot", actions=["read_file"])
        tools = ToolRegistry()
        tools.add("read_file", lambda path: f"contents of {path}")

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = govern_tool_call(
            executor, "read_file", {"path": "test.txt"}, tools
        )
        assert result == "contents of test.txt"

    def test_convenience_function_denied(self, tmp_path):
        """govern_tool_call handles denials."""
        _setup_agent(tmp_path, "TestBot", actions=["read"])
        tools = ToolRegistry()
        tools.add("delete_file", lambda path: None)

        executor = GovernedToolExecutor.connect("TestBot", base_dir=tmp_path)

        result = govern_tool_call(
            executor, "delete_file", {"path": "x"}, tools
        )
        assert "denied" in result.lower()


class TestImports:
    def test_import_from_integrations(self):
        """All public classes are importable from the integrations module."""
        from nomotic.integrations.tool_adapter import (
            GovernanceDeniedError,
            ToolAdapter,
            ToolDefinition,
            ToolRegistry,
            govern_tool_call,
        )

        assert ToolRegistry is not None
        assert ToolAdapter is not None
        assert GovernanceDeniedError is not None
        assert ToolDefinition is not None
        assert govern_tool_call is not None

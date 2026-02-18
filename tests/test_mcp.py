"""Tests for MCP integration."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.mcp.client import MCPClient
from nanobot.mcp.config_loader import MCPConfigLoader
from nanobot.mcp.registry import MCPRegistry
from nanobot.mcp.schemas import (
    MCPExecutionResponse,
    MCPServerConfig,
    MCPToolSchema,
)
from nanobot.mcp.tool_adapter import MCPToolAdapter


# Test fixtures
@pytest.fixture
def sample_server_config():
    """Create a sample server configuration."""
    return MCPServerConfig(
        name="test_server",
        type="local",
        command="python",
        args=["-m", "test_module"],
        env={"TEST_VAR": "test_value"},
    )


@pytest.fixture
def sample_tool_schema():
    """Create a sample tool schema."""
    return MCPToolSchema(
        name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    )


# Test MCPToolSchema
def test_mcp_tool_schema_validation():
    """Test MCPToolSchema validation."""
    # Valid schema
    schema = MCPToolSchema(
        name="valid_tool",
        description="Valid tool",
        input_schema={"type": "object"},
    )
    assert schema.name == "valid_tool"
    assert schema.description == "Valid tool"
    assert schema.input_schema == {"type": "object"}

    # Invalid: empty name
    with pytest.raises(ValueError, match="name cannot be empty"):
        MCPToolSchema(
            name="",
            description="Test",
            input_schema={},
        )

    # Invalid: non-dict schema
    with pytest.raises(ValueError, match="must be a dictionary"):
        MCPToolSchema(
            name="test",
            description="Test",
            input_schema="invalid",  # type: ignore
        )


# Test MCPExecutionResponse
def test_mcp_execution_response_to_string():
    """Test converting execution response to string."""
    # Success response
    response = MCPExecutionResponse(
        content=[
            {"text": "Line 1"},
            {"text": "Line 2"},
        ],
        is_error=False,
    )
    assert response.to_string() == "Line 1\nLine 2"

    # Error response
    error_response = MCPExecutionResponse(
        content=[{"text": "Something went wrong"}],
        is_error=True,
    )
    assert error_response.to_string() == "Error: Something went wrong"

    # Empty response
    empty_response = MCPExecutionResponse(content=[], is_error=False)
    assert empty_response.to_string() == ""


# Test MCPConfigLoader
def test_mcp_config_loader_validation():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        "name": "test",
        "command": "python",
        "type": "local",
        "args": ["arg1", "arg2"],
        "env": {"KEY": "value"},
    }
    assert MCPConfigLoader.validate(valid_config) is True

    # Missing name
    assert MCPConfigLoader.validate({"command": "python"}) is False

    # Missing command
    assert MCPConfigLoader.validate({"name": "test"}) is False

    # Invalid type
    assert MCPConfigLoader.validate({
        "name": "test",
        "command": "python",
        "type": "invalid",
    }) is False

    # Invalid args (not list)
    assert MCPConfigLoader.validate({
        "name": "test",
        "command": "python",
        "args": "not a list",
    }) is False


def test_mcp_config_loader_from_yaml(tmp_path: Path):
    """Test loading configuration from YAML file."""
    # Create test config file
    config_file = tmp_path / "mcp.yaml"
    config_file.write_text("""
mcp_servers:
  - name: server1
    command: python
    args: ["-m", "module1"]
    env:
      KEY1: value1

  - name: server2
    command: node
    type: local
""")

    # Load configs
    configs = MCPConfigLoader.load(config_file)

    assert len(configs) == 2
    assert configs[0].name == "server1"
    assert configs[0].command == "python"
    assert configs[0].args == ["-m", "module1"]
    assert configs[0].env == {"KEY1": "value1"}

    assert configs[1].name == "server2"
    assert configs[1].command == "node"
    assert configs[1].type == "local"


def test_mcp_config_loader_invalid_yaml(tmp_path: Path):
    """Test handling of invalid YAML."""
    config_file = tmp_path / "mcp.yaml"
    config_file.write_text("invalid: yaml: content:")

    with pytest.raises(ValueError, match="Invalid YAML"):
        MCPConfigLoader.load(config_file)


def test_mcp_config_loader_missing_file(tmp_path: Path):
    """Test handling of missing config file."""
    missing_file = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        MCPConfigLoader.load(missing_file)


# Test MCPClient (mocked)
@pytest.mark.asyncio
async def test_mcp_client_connect_and_discover(sample_server_config):
    """Test client connection and tool discovery."""
    client = MCPClient(sample_server_config)

    # Mock the subprocess
    mock_process = AsyncMock()
    mock_process.stdin = AsyncMock()
    mock_process.stdout = AsyncMock()
    mock_process.stderr = AsyncMock()

    # Mock initialization response
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
        },
    }

    # Mock tools/list response
    tools_response = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {"type": "object"},
                }
            ],
        },
    }

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Mock responses
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                json.dumps(init_response).encode() + b"\n",
                json.dumps(tools_response).encode() + b"\n",
            ]
        )

        # Connect
        await client.connect()
        assert client.is_connected()

        # Discover tools
        tools = await client.discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

        # Cleanup
        await client.disconnect()


@pytest.mark.asyncio
async def test_mcp_client_execute_tool(sample_server_config):
    """Test tool execution."""
    client = MCPClient(sample_server_config)
    client._connected = True

    # Mock process
    mock_process = AsyncMock()
    mock_process.stdin = AsyncMock()
    mock_process.stdout = AsyncMock()
    client.process = mock_process

    # Mock tool execution response
    exec_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"text": "Tool executed successfully"},
            ],
        },
    }

    mock_process.stdout.readline = AsyncMock(
        return_value=json.dumps(exec_response).encode() + b"\n"
    )

    # Execute tool
    response = await client.execute_tool("test_tool", {"query": "test"})

    assert not response.is_error
    assert "successfully" in response.to_string()


# Test MCPToolAdapter
@pytest.mark.asyncio
async def test_mcp_tool_adapter(sample_server_config, sample_tool_schema):
    """Test the tool adapter."""
    # Create mock client
    mock_client = Mock(spec=MCPClient)
    mock_client.config = sample_server_config
    mock_client.is_connected.return_value = True
    mock_client.execute_tool = AsyncMock(
        return_value=MCPExecutionResponse(
            content=[{"text": "Result from MCP tool"}],
            is_error=False,
        )
    )

    # Create adapter
    adapter = MCPToolAdapter(mock_client, sample_tool_schema)

    # Test properties
    assert "test_server" in adapter.name
    assert "test_tool" in adapter.name
    assert "test_server" in adapter.description
    assert adapter.parameters["type"] == "object"

    # Test execution
    result = await adapter.execute(query="test query")
    assert "Result from MCP tool" in result

    # Verify client was called
    mock_client.execute_tool.assert_called_once_with(
        "test_tool",
        {"query": "test query"}
    )


@pytest.mark.asyncio
async def test_mcp_tool_adapter_disconnected_client(sample_server_config, sample_tool_schema):
    """Test adapter behavior when client is disconnected."""
    mock_client = Mock(spec=MCPClient)
    mock_client.config = sample_server_config
    mock_client.is_connected.return_value = False

    adapter = MCPToolAdapter(mock_client, sample_tool_schema)
    result = await adapter.execute(query="test")

    assert "not connected" in result


# Test MCPRegistry
@pytest.mark.asyncio
async def test_mcp_registry_add_client(sample_server_config):
    """Test adding an MCP client to the registry."""
    tool_registry = ToolRegistry()
    mcp_registry = MCPRegistry(tool_registry)

    # Mock client
    mock_client = AsyncMock(spec=MCPClient)
    mock_client.config = sample_server_config
    mock_client.discover_tools = AsyncMock(return_value=[
        MCPToolSchema(
            name="tool1",
            description="Tool 1",
            input_schema={"type": "object"},
        ),
        MCPToolSchema(
            name="tool2",
            description="Tool 2",
            input_schema={"type": "object"},
        ),
    ])

    with patch("nanobot.mcp.registry.MCPClient", return_value=mock_client):
        await mcp_registry.add_client(sample_server_config)

    # Verify client was added
    assert "test_server" in mcp_registry.client_names

    # Verify tools were registered
    assert len(mcp_registry._registered_tools["test_server"]) == 2

    # Verify tools are in tool registry
    assert len(tool_registry) >= 2


@pytest.mark.asyncio
async def test_mcp_registry_unregister_tools(sample_server_config):
    """Test unregistering tools from a client."""
    tool_registry = ToolRegistry()
    mcp_registry = MCPRegistry(tool_registry)

    # Mock some registered tools
    mcp_registry._registered_tools["test_server"] = [
        "mcp_test_server_tool1",
        "mcp_test_server_tool2",
    ]

    # Add tools to registry
    from nanobot.agent.tools.base import Tool

    class MockTool(Tool):
        def __init__(self, name: str):
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return "Mock tool"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object"}

        async def execute(self, **kwargs: Any) -> str:
            return "mock"

    tool_registry.register(MockTool("mcp_test_server_tool1"))
    tool_registry.register(MockTool("mcp_test_server_tool2"))

    initial_count = len(tool_registry)

    # Unregister
    mcp_registry.unregister_tools("test_server")

    # Verify tools were removed
    assert len(tool_registry) == initial_count - 2
    assert "test_server" not in mcp_registry._registered_tools


@pytest.mark.asyncio
async def test_mcp_registry_disconnect_all(sample_server_config):
    """Test disconnecting all clients."""
    tool_registry = ToolRegistry()
    mcp_registry = MCPRegistry(tool_registry)

    # Add mock clients
    mock_client1 = AsyncMock(spec=MCPClient)
    mock_client1.config = sample_server_config
    mock_client1.disconnect = AsyncMock()

    mock_client2 = AsyncMock(spec=MCPClient)
    mock_client2.config = MCPServerConfig(
        name="server2",
        type="local",
        command="test",
    )
    mock_client2.disconnect = AsyncMock()

    mcp_registry._clients["server1"] = mock_client1
    mcp_registry._clients["server2"] = mock_client2
    mcp_registry._registered_tools["server1"] = []
    mcp_registry._registered_tools["server2"] = []

    # Disconnect all
    await mcp_registry.disconnect_all()

    # Verify disconnects were called
    mock_client1.disconnect.assert_called_once()
    mock_client2.disconnect.assert_called_once()

    # Verify clients were removed
    assert len(mcp_registry.client_names) == 0


# Integration test
@pytest.mark.asyncio
async def test_integration_with_tool_registry():
    """Test full integration with ToolRegistry."""
    tool_registry = ToolRegistry()

    # Create mock MCP client with tools
    mock_client = AsyncMock(spec=MCPClient)
    mock_client.config = MCPServerConfig(
        name="integration_test",
        type="local",
        command="test",
    )
    mock_client.is_connected.return_value = True
    mock_client.discover_tools = AsyncMock(return_value=[
        MCPToolSchema(
            name="echo",
            description="Echo tool",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        ),
    ])
    mock_client.execute_tool = AsyncMock(
        return_value=MCPExecutionResponse(
            content=[{"text": "Echo: test message"}],
            is_error=False,
        )
    )

    # Register with MCP registry
    mcp_registry = MCPRegistry(tool_registry)

    with patch("nanobot.mcp.registry.MCPClient", return_value=mock_client):
        await mcp_registry.add_client(mock_client.config)

    # Verify tool is available in registry
    tools = tool_registry.get_definitions()
    tool_names = [t["function"]["name"] for t in tools]
    assert any("echo" in name for name in tool_names)

    # Execute tool through registry
    tool_name = [n for n in tool_names if "echo" in n][0]
    result = await tool_registry.execute(tool_name, {"message": "test message"})

    assert "Echo" in result
    assert "test message" in result

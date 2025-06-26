
# MCP (Model Context Protocol) Demonstration

This directory contains a comprehensive demonstration of how MCP (Model Context Protocol) works, showing how AI assistants like Claude interact with external tools and resources through a standardized protocol.

## Files

- `mcp_demo.py` - Main demonstration script
- `mcp_config.json` - Configuration file for MCP demo
- `README_MCP.md` - This documentation file

## What is MCP?

Model Context Protocol (MCP) is a standardized way for AI assistants to securely interact with external tools and resources. Instead of AI assistants having direct access to your system, MCP provides:

1. **Structured Communication**: Standardized message format for tool calls
2. **Security**: Controlled access through well-defined interfaces
3. **Discoverability**: Tools and resources can describe their capabilities
4. **Validation**: Input/output schemas ensure data integrity

## Demo Features

### 🔧 Tool Demonstrations
- **File Operations**: Reading files from the filesystem
- **Web Search**: Searching for information online
- **Code Execution**: Running Python code safely
- **Data Analysis**: Processing and analyzing datasets

### 📄 Resource Access
- **Configuration**: Reading application settings
- **Documentation**: Accessing API documentation
- **Data**: Retrieving example datasets

### 🚀 Advanced Scenarios
- **Multi-step Workflows**: Chaining multiple tool calls
- **Error Handling**: Graceful error recovery
- **Interactive Mode**: User-driven tool exploration

## Running the Demo

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install rich asyncio requests
```

### Basic Usage

```bash
cd extras
python mcp_demo.py
```

### What You'll See

1. **Server Connection**: Simulated MCP handshake process
2. **Capability Discovery**: Listing available tools and resources
3. **Tool Usage Examples**: Practical demonstrations of each tool
4. **Resource Access**: Reading different types of resources
5. **Advanced Scenarios**: Complex workflows and error handling
6. **Interactive Mode**: Try tools yourself with custom inputs

## Understanding the Code

### Key Classes

#### `MCPServer`
Represents an MCP server that provides tools and resources. In reality, this would be a separate service (like a file system interface, database connector, or API gateway).

```python
server = MCPServer("Workshop Tools Server")
# Provides tools like read_file, web_search, execute_python
# Provides resources like configuration, documentation, data
```

#### `MCPClient`
Represents an AI assistant (like Claude) that uses MCP to access tools and resources.

```python
client = MCPClient("Claude AI Assistant")
client.connect_server(server)
result = await client.call_tool("server_name", "tool_name", arguments)
```

#### `MCPMessage`
Standardized message format for MCP communication.

```python
message = MCPMessage(
    id="msg_123",
    method="tools/call",
    params={"name": "read_file", "arguments": {"path": "example.txt"}}
)
```

### Message Flow

1. **Discovery**: Client asks server what tools/resources are available
2. **Tool Call**: Client sends structured request to use a tool
3. **Execution**: Server safely executes the tool
4. **Response**: Server returns structured result
5. **Processing**: Client processes the result

## Real-World Applications

### Development Tools
- Code editors calling language servers
- IDEs integrating with build systems
- Debug tools accessing runtime information

### Data Processing
- AI assistants querying databases
- Automated report generation
- Data pipeline orchestration

### System Integration
- Cloud service management
- File system operations
- API integrations

## Security Benefits

### Controlled Access
- Tools define exactly what they can do
- Input validation prevents malicious requests
- Output schemas ensure data integrity

### Audit Trail
- All tool calls are logged
- Easy to track what AI assistants are doing
- Security monitoring and compliance

### Sandboxing
- Tools run in controlled environments
- Limited system access
- Safe code execution

## Learning Exercise

Try modifying the demo to:

1. **Add New Tools**: Create your own tool (e.g., image processing, email sending)
2. **Custom Resources**: Add new resource types (e.g., database connections)
3. **Error Scenarios**: Test different error conditions
4. **Performance**: Add timing and performance metrics
5. **Security**: Implement access controls and validation

## Next Steps

After running this demo:

1. **Explore Real MCP**: Look at actual MCP implementations
2. **Build Tools**: Create your own MCP-compatible tools
3. **Integration**: Connect real AI assistants to your tools
4. **Production**: Deploy MCP servers for real applications

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Claude MCP Documentation](https://docs.anthropic.com/en/docs/build-with-claude/computer-use)

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install rich requests
```

**Async Errors**
- Ensure you're using Python 3.7+ for async/await support

**Permission Errors**
- The demo runs in simulation mode, so no actual file system access is needed

### Demo Not Interactive?
- Make sure your terminal supports interactive input
- Try running with `python -u mcp_demo.py` for unbuffered output

---

This demo provides a comprehensive understanding of MCP concepts through hands-on exploration. Experiment with the code to deepen your understanding!

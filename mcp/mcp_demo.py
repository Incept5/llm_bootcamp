
#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Demonstration Script

This script provides a comprehensive demonstration of how MCP works by simulating
the protocol interactions between AI assistants (like Claude) and external tools/resources.

Author: Workshop Instructor
Date: 2025-01-24
"""

import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import os
from pathlib import Path

# Rich library for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm

# Set up rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("mcp_demo")


class MCPMessageType(Enum):
    """MCP Protocol Message Types"""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPMessage:
    """Represents an MCP protocol message"""
    id: str
    method: str
    params: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MCPResource:
    """Represents an MCP resource"""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    
    def to_dict(self):
        return asdict(self)


class MCPServer:
    """
    Simulated MCP Server
    
    In reality, this would be a separate process/service that provides
    tools and resources to AI assistants through the MCP protocol.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self._setup_default_tools()
        self._setup_default_resources()
    
    def _setup_default_tools(self):
        """Set up default tools that this server provides"""
        
        # File operations tool
        self.tools.append(MCPTool(
            name="read_file",
            description="Read contents of a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            }
        ))
        
        # Web search tool
        self.tools.append(MCPTool(
            name="web_search",
            description="Search the web for information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ))
        
        # Code execution tool
        self.tools.append(MCPTool(
            name="execute_python",
            description="Execute Python extras and return results",
            input_schema={
                "type": "object",
                "properties": {
                    "extras": {
                        "type": "string",
                        "description": "Python extras to execute"
                    }
                },
                "required": ["extras"]
            }
        ))
        
        # Data analysis tool
        self.tools.append(MCPTool(
            name="analyze_data",
            description="Analyze data from CSV files or datasets",
            input_schema={
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to data file"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["summary", "correlation", "visualization"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["data_path", "analysis_type"]
            }
        ))
    
    def _setup_default_resources(self):
        """Set up default resources that this server provides"""
        
        # Configuration resource
        self.resources.append(MCPResource(
            uri="config://app/settings",
            name="Application Settings",
            description="Current application configuration",
            mime_type="application/json"
        ))
        
        # Documentation resource
        self.resources.append(MCPResource(
            uri="docs://api/reference",
            name="API Reference",
            description="Complete API documentation",
            mime_type="text/markdown"
        ))
        
        # Data resource
        self.resources.append(MCPResource(
            uri="data://workshop/examples",
            name="Workshop Examples",
            description="Example datasets and files used in the workshop",
            mime_type="application/json"
        ))
    
    async def handle_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle incoming MCP messages"""
        
        logger.info(f"[{self.name}] Received: {message.method}")
        
        if message.method == MCPMessageType.LIST_TOOLS.value:
            return {
                "tools": [tool.to_dict() for tool in self.tools]
            }
        
        elif message.method == MCPMessageType.CALL_TOOL.value:
            return await self._execute_tool(message.params)
        
        elif message.method == MCPMessageType.LIST_RESOURCES.value:
            return {
                "resources": [resource.to_dict() for resource in self.resources]
            }
        
        elif message.method == MCPMessageType.READ_RESOURCE.value:
            return await self._read_resource(message.params)
        
        else:
            return {"error": f"Unknown method: {message.method}"}
    
    async def _execute_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool based on the provided parameters"""
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        console.print(f"🔧 Executing tool: [bold]{tool_name}[/bold]")
        
        # Simulate some processing time
        await asyncio.sleep(0.5)
        
        if tool_name == "read_file":
            return await self._tool_read_file(arguments)
        elif tool_name == "web_search":
            return await self._tool_web_search(arguments)
        elif tool_name == "execute_python":
            return await self._tool_execute_python(arguments)
        elif tool_name == "analyze_data":
            return await self._tool_analyze_data(arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _tool_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate reading a file"""
        file_path = args.get("path")
        
        try:
            # For demo purposes, we'll handle some special cases
            if file_path == "README.md":
                content = "# MCP Demo Project\n\nThis demonstrates Model Context Protocol usage."
            elif file_path == "config.json":
                content = json.dumps({
                    "app_name": "MCP Demo",
                    "version": "1.0.0",
                    "features": ["tools", "resources", "async_processing"]
                }, indent=2)
            elif file_path.endswith(".py"):
                content = "# Python extras example\nprint('Hello from MCP!')\nresult = 42\n"
            else:
                # Try to read actual file if it exists
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    return {"error": f"File not found: {file_path}"}
            
            return {
                "content": content,
                "metadata": {
                    "file_path": file_path,
                    "size": len(content),
                    "encoding": "utf-8"
                }
            }
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    async def _tool_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate web search"""
        query = args.get("query")
        max_results = args.get("max_results", 5)
        
        # Simulate search results
        results = [
            {
                "title": f"Understanding {query} - Comprehensive Guide",
                "url": f"https://example.com/{query.lower().replace(' ', '-')}",
                "snippet": f"Learn everything about {query} with this detailed guide covering all aspects...",
                "relevance_score": 0.95
            },
            {
                "title": f"{query} Best Practices and Examples",
                "url": f"https://docs.example.com/{query.lower()}",
                "snippet": f"Discover best practices for implementing {query} in real-world applications...",
                "relevance_score": 0.87
            },
            {
                "title": f"Advanced {query} Techniques",
                "url": f"https://advanced.example.com/{query.lower()}",
                "snippet": f"Master advanced techniques and patterns for {query} implementation...",
                "relevance_score": 0.82
            }
        ]
        
        return {
            "query": query,
            "results": results[:max_results],
            "total_found": len(results),
            "search_time_ms": 245
        }
    
    async def _tool_execute_python(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Python extras execution"""
        code = args.get("extras")
        
        # For safety, we'll simulate execution rather than actually executing
        # In a real implementation, this would use a sandboxed environment
        
        simulated_outputs = {
            "print('Hello, MCP!')": "Hello, MCP!",
            "2 + 2": "4",
            "import math; math.pi": "3.141592653589793",
            "list(range(5))": "[0, 1, 2, 3, 4]",
            "[x**2 for x in range(5)]": "[0, 1, 4, 9, 16]"
        }
        
        # Try to find a matching simulated output
        output = simulated_outputs.get(code.strip(), f"# Executed: {code}\n# Output: <simulated result>")
        
        return {
            "extras": code,
            "output": output,
            "execution_time_ms": 123,
            "status": "success"
        }
    
    async def _tool_analyze_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data analysis"""
        data_path = args.get("data_path")
        analysis_type = args.get("analysis_type")
        
        # Simulate analysis results
        if analysis_type == "summary":
            result = {
                "rows": 1000,
                "columns": 5,
                "numeric_columns": 3,
                "categorical_columns": 2,
                "missing_values": 25,
                "memory_usage": "78.5 KB"
            }
        elif analysis_type == "correlation":
            result = {
                "correlations": {
                    "feature1_vs_feature2": 0.85,
                    "feature1_vs_feature3": -0.23,
                    "feature2_vs_feature3": 0.67
                },
                "strong_correlations": ["feature1_vs_feature2", "feature2_vs_feature3"]
            }
        elif analysis_type == "visualization":
            result = {
                "charts_generated": ["histogram", "scatter_plot", "correlation_matrix"],
                "output_path": f"/tmp/analysis_{int(time.time())}.html",
                "chart_count": 3
            }
        else:
            result = {"error": f"Unknown analysis type: {analysis_type}"}
        
        return {
            "data_path": data_path,
            "analysis_type": analysis_type,
            "result": result,
            "processing_time_ms": 1500
        }
    
    async def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a resource based on URI"""
        uri = params.get("uri")
        
        # Simulate resource content based on URI
        if uri == "config://app/settings":
            content = json.dumps({
                "mcp_version": "2024-11-05",
                "server_capabilities": ["tools", "resources", "async"],
                "max_concurrent_requests": 10,
                "timeout_seconds": 30
            }, indent=2)
        elif uri == "docs://api/reference":
            content = """# MCP API Reference

## Tools
- `read_file`: Read file contents
- `web_search`: Search the web
- `execute_python`: Execute Python extras
- `analyze_data`: Analyze datasets

## Resources
- Configuration settings
- API documentation
- Example datasets
"""
        elif uri == "data://workshop/examples":
            content = json.dumps({
                "examples": [
                    {"name": "Alice in Wonderland", "path": "demos/Alice_in_Wonderland.txt"},
                    {"name": "Grimm Fairy Tales", "path": "demos/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt"},
                    {"name": "Workshop Code", "path": "extras/"}
                ]
            }, indent=2)
        else:
            return {"error": f"Resource not found: {uri}"}
        
        return {
            "uri": uri,
            "content": content,
            "metadata": {
                "size": len(content),
                "last_modified": datetime.now().isoformat(),
                "content_type": "application/json" if uri.endswith("settings") else "text/plain"
            }
        }


class MCPClient:
    """
    Simulated MCP Client (like Claude or another AI assistant)
    
    This represents how an AI assistant would interact with MCP servers
    to access tools and resources.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.servers: Dict[str, MCPServer] = {}
        self.message_id = 0
    
    def connect_server(self, server: MCPServer):
        """Connect to an MCP server"""
        self.servers[server.name] = server
        console.print(f"🔗 Connected to MCP server: [bold]{server.name}[/bold]")
    
    def _next_message_id(self) -> str:
        """Generate next message ID"""
        self.message_id += 1
        return f"msg_{self.message_id}"
    
    async def list_available_tools(self) -> Dict[str, List[MCPTool]]:
        """List all available tools from connected servers"""
        all_tools = {}
        
        for server_name, server in self.servers.items():
            message = MCPMessage(
                id=self._next_message_id(),
                method=MCPMessageType.LIST_TOOLS.value,
                params={}
            )
            
            response = await server.handle_message(message)
            tools = [MCPTool(**tool_data) for tool_data in response.get("tools", [])]
            all_tools[server_name] = tools
        
        return all_tools
    
    async def list_available_resources(self) -> Dict[str, List[MCPResource]]:
        """List all available resources from connected servers"""
        all_resources = {}
        
        for server_name, server in self.servers.items():
            message = MCPMessage(
                id=self._next_message_id(),
                method=MCPMessageType.LIST_RESOURCES.value,
                params={}
            )
            
            response = await server.handle_message(message)
            resources = [MCPResource(**res_data) for res_data in response.get("resources", [])]
            all_resources[server_name] = resources
        
        return all_resources
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool on a server"""
        if server_name not in self.servers:
            return {"error": f"Server not found: {server_name}"}
        
        server = self.servers[server_name]
        message = MCPMessage(
            id=self._next_message_id(),
            method=MCPMessageType.CALL_TOOL.value,
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        return await server.handle_message(message)
    
    async def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Read a resource from a server"""
        if server_name not in self.servers:
            return {"error": f"Server not found: {server_name}"}
        
        server = self.servers[server_name]
        message = MCPMessage(
            id=self._next_message_id(),
            method=MCPMessageType.READ_RESOURCE.value,
            params={"uri": uri}
        )
        
        return await server.handle_message(message)


class MCPDemo:
    """Main demo class that orchestrates the MCP demonstration"""
    
    def __init__(self):
        self.client = MCPClient("Claude AI Assistant")
        self.server = MCPServer("Workshop Tools Server")
        
    async def run_full_demo(self):
        """Run the complete MCP demonstration"""
        
        console.print(Panel.fit(
            "[bold blue]🚀 Model Context Protocol (MCP) Demo[/bold blue]\n\n"
            "This demonstration shows how AI assistants like Claude use MCP\n"
            "to access external tools and resources safely and efficiently.",
            title="Welcome to MCP Demo"
        ))
        
        # Step 1: Server Connection
        await self._demo_server_connection()
        
        # Step 2: Capability Discovery
        await self._demo_capability_discovery()
        
        # Step 3: Tool Usage Examples
        await self._demo_tool_usage()
        
        # Step 4: Resource Access Examples
        await self._demo_resource_access()
        
        # Step 5: Advanced Scenarios
        await self._demo_advanced_scenarios()
        
        # Step 6: Interactive Mode
        if Confirm.ask("\n🎮 Would you like to try interactive mode?"):
            await self._interactive_mode()
        
        console.print(Panel.fit(
            "[bold green]✅ MCP Demo Complete![/bold green]\n\n"
            "You've seen how MCP enables secure, structured communication\n"
            "between AI assistants and external tools/resources.",
            title="Demo Complete"
        ))
    
    async def _demo_server_connection(self):
        """Demonstrate MCP server connection process"""
        
        console.print("\n[bold]Step 1: Server Connection[/bold]")
        console.print("Connecting AI assistant to MCP server...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Establishing connection...", total=None)
            await asyncio.sleep(1)
            
            progress.update(task, description="Performing handshake...")
            await asyncio.sleep(1)
            
            progress.update(task, description="Negotiating capabilities...")
            await asyncio.sleep(1)
            
            progress.update(task, description="Connection established!")
            await asyncio.sleep(0.5)
        
        self.client.connect_server(self.server)
        console.print("✅ Successfully connected to MCP server\n")
    
    async def _demo_capability_discovery(self):
        """Demonstrate discovering available tools and resources"""
        
        console.print("[bold]Step 2: Capability Discovery[/bold]")
        console.print("Discovering available tools and resources...\n")
        
        # List tools
        tools = await self.client.list_available_tools()
        
        console.print("[bold cyan]Available Tools:[/bold cyan]")
        for server_name, server_tools in tools.items():
            tool_tree = Tree(f"📡 {server_name}")
            for tool in server_tools:
                tool_tree.add(f"🔧 {tool.name}: {tool.description}")
            console.print(tool_tree)
        
        # List resources
        resources = await self.client.list_available_resources()
        
        console.print("\n[bold cyan]Available Resources:[/bold cyan]")
        for server_name, server_resources in resources.items():
            resource_tree = Tree(f"📡 {server_name}")
            for resource in server_resources:
                resource_tree.add(f"📄 {resource.name}: {resource.description}")
            console.print(resource_tree)
        
        console.print()
    
    async def _demo_tool_usage(self):
        """Demonstrate using various MCP tools"""
        
        console.print("[bold]Step 3: Tool Usage Examples[/bold]")
        
        # Example 1: File reading
        console.print("\n[cyan]Example 1: Reading a file[/cyan]")
        result = await self.client.call_tool(
            "Workshop Tools Server",
            "read_file",
            {"path": "README.md"}
        )
        
        if "error" not in result:
            console.print("📄 File contents preview:")
            content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
            console.print(Panel(content_preview, title="README.md"))
        
        # Example 2: Web search
        console.print("\n[cyan]Example 2: Web search[/cyan]")
        result = await self.client.call_tool(
            "Workshop Tools Server",
            "web_search",
            {"query": "MCP Model Context Protocol", "max_results": 3}
        )
        
        if "error" not in result:
            console.print("🔍 Search results:")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Title", style="cyan")
            table.add_column("URL", style="blue")
            table.add_column("Relevance", style="green")
            
            for search_result in result.get("results", []):
                table.add_row(
                    search_result["title"][:50] + "...",
                    search_result["url"][:40] + "...",
                    f"{search_result['relevance_score']:.2f}"
                )
            console.print(table)
        
        # Example 3: Code execution
        console.print("\n[cyan]Example 3: Python extras execution[/cyan]")
        result = await self.client.call_tool(
            "Workshop Tools Server",
            "execute_python",
            {"extras": "print('Hello from MCP!'); result = 2 + 2; print(f'2 + 2 = {result}')"}
        )
        
        if "error" not in result:
            console.print("🐍 Execution result:")
            console.print(Panel(result["output"], title="Python Output"))
        
        # Example 4: Data analysis
        console.print("\n[cyan]Example 4: Data analysis[/cyan]")
        result = await self.client.call_tool(
            "Workshop Tools Server",
            "analyze_data",
            {"data_path": "sample_data.csv", "analysis_type": "summary"}
        )
        
        if "error" not in result:
            console.print("📊 Analysis result:")
            analysis_json = json.dumps(result["result"], indent=2)
            console.print(Panel(analysis_json, title="Data Summary"))
    
    async def _demo_resource_access(self):
        """Demonstrate accessing MCP resources"""
        
        console.print("\n[bold]Step 4: Resource Access Examples[/bold]")
        
        resources = [
            ("config://app/settings", "Application Configuration"),
            ("docs://api/reference", "API Documentation"),
            ("data://workshop/examples", "Workshop Examples")
        ]
        
        for uri, description in resources:
            console.print(f"\n[cyan]{description}[/cyan]")
            result = await self.client.read_resource("Workshop Tools Server", uri)
            
            if "error" not in result:
                content = result["content"]
                if uri.endswith("settings") or uri.endswith("examples"):
                    # JSON content - pretty print
                    syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=f"Resource: {uri}"))
                else:
                    # Markdown content
                    syntax = Syntax(content, "markdown", theme="monokai")
                    console.print(Panel(syntax, title=f"Resource: {uri}"))
    
    async def _demo_advanced_scenarios(self):
        """Demonstrate advanced MCP usage scenarios"""
        
        console.print("\n[bold]Step 5: Advanced Scenarios[/bold]")
        
        # Scenario 1: Multi-step workflow
        console.print("\n[cyan]Scenario 1: Multi-step data processing workflow[/cyan]")
        console.print("1. Read configuration → 2. Analyze data → 3. Generate report")
        
        # Step 1: Read config
        config_result = await self.client.read_resource(
            "Workshop Tools Server", 
            "config://app/settings"
        )
        
        # Step 2: Analyze data
        analysis_result = await self.client.call_tool(
            "Workshop Tools Server",
            "analyze_data",
            {"data_path": "workshop_data.csv", "analysis_type": "correlation"}
        )
        
        # Step 3: Generate summary
        summary_code = """
import json
config = """ + config_result.get("content", "{}") + """
analysis = """ + json.dumps(analysis_result.get("result", {})) + """
print("Workflow Summary:")
print(f"- Config version: {json.loads(config).get('mcp_version', 'unknown')}")
print(f"- Analysis type: correlation")
print(f"- Strong correlations found: {len(analysis.get('strong_correlations', []))}")
"""
        
        exec_result = await self.client.call_tool(
            "Workshop Tools Server",
            "execute_python",
            {"extras": summary_code}
        )
        
        console.print("🔄 Workflow completed:")
        console.print(Panel(exec_result.get("output", ""), title="Workflow Summary"))
        
        # Scenario 2: Error handling
        console.print("\n[cyan]Scenario 2: Error handling and recovery[/cyan]")
        
        # Try to read non-existent file
        error_result = await self.client.call_tool(
            "Workshop Tools Server",
            "read_file",
            {"path": "non_existent_file.txt"}
        )
        
        if "error" in error_result:
            console.print("❌ Expected error occurred:")
            console.print(Panel(error_result["error"], title="Error Message", border_style="red"))
            
            # Recovery: read a file that exists
            console.print("🔄 Recovering by reading an existing file...")
            recovery_result = await self.client.call_tool(
                "Workshop Tools Server",
                "read_file",
                {"path": "config.json"}
            )
            
            if "error" not in recovery_result:
                console.print("✅ Recovery successful!")
                console.print(Panel(recovery_result["content"], title="Recovery Result"))
    
    async def _interactive_mode(self):
        """Interactive mode for user experimentation"""
        
        console.print("\n[bold]🎮 Interactive Mode[/bold]")
        console.print("Try calling MCP tools yourself!\n")
        
        # Show available tools
        tools = await self.client.list_available_tools()
        tool_names = []
        for server_tools in tools.values():
            tool_names.extend([tool.name for tool in server_tools])
        
        console.print("Available tools:", ", ".join(tool_names))
        
        while True:
            try:
                tool_name = Prompt.ask("\nWhich tool would you like to use? (or 'quit' to exit)")
                
                if tool_name.lower() in ['quit', 'exit', 'q']:
                    break
                
                if tool_name not in tool_names:
                    console.print(f"❌ Unknown tool: {tool_name}")
                    continue
                
                # Get tool schema and prompt for arguments
                console.print(f"\n[cyan]Using tool: {tool_name}[/cyan]")
                
                if tool_name == "read_file":
                    path = Prompt.ask("File path")
                    result = await self.client.call_tool("Workshop Tools Server", tool_name, {"path": path})
                
                elif tool_name == "web_search":
                    query = Prompt.ask("Search query")
                    max_results = Prompt.ask("Max results", default="5")
                    result = await self.client.call_tool("Workshop Tools Server", tool_name, {
                        "query": query, 
                        "max_results": int(max_results)
                    })
                
                elif tool_name == "execute_python":
                    code = Prompt.ask("Python extras")
                    result = await self.client.call_tool("Workshop Tools Server", tool_name, {"extras": code})
                
                elif tool_name == "analyze_data":
                    data_path = Prompt.ask("Data path")
                    analysis_type = Prompt.ask("Analysis type", choices=["summary", "correlation", "visualization"])
                    result = await self.client.call_tool("Workshop Tools Server", tool_name, {
                        "data_path": data_path,
                        "analysis_type": analysis_type
                    })
                
                # Display result
                if "error" in result:
                    console.print(Panel(result["error"], title="Error", border_style="red"))
                else:
                    result_json = json.dumps(result, indent=2)
                    console.print(Panel(result_json, title="Tool Result"))
                
            except KeyboardInterrupt:
                console.print("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}")


async def main():
    """Main function to run the MCP demo"""
    try:
        demo = MCPDemo()
        await demo.run_full_demo()
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"❌ Demo failed: {str(e)}")
        logger.exception("Demo error")


if __name__ == "__main__":
    console.print("[bold]🚀 Starting MCP Demo...[/bold]")
    asyncio.run(main())

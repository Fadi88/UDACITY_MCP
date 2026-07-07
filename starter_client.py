import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, List, Dict, TypedDict
from datetime import datetime, timedelta
from pathlib import Path
import re

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str | Path) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
            ValueError: If configuration file is missing required fields.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e.msg}", e.doc, e.pos)
            
        if "mcpServers" not in config:
            raise ValueError("Configuration missing required 'mcpServers' field")
            
        return config

    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})}
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            logging.info(f"✓ Server '{self.name}' initialized")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the server.

        Returns:
            A list of available tool definitions.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")
            
        result = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} is not initialized")

        for attempt in range(retries + 1):
            try:
                logging.info(f"Attempting execution of tool '{tool_name}' (attempt {attempt + 1}/{retries + 1})")
                result = await self.session.call_tool(
                    tool_name, 
                    arguments, 
                    read_timeout_seconds=timedelta(seconds=60)
                )
                return result
            except Exception as e:
                if attempt == retries:
                    logging.error(f"Tool execution failed after {retries} retries: {e}")
                    raise
                logging.warning(f"Tool execution failed, retrying in {delay}s... ({e})")
                await asyncio.sleep(delay)

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class DataExtractor:
    """Handles extraction and storage of structured data from LLM responses."""
    
    def __init__(self, sqlite_server: Server, anthropic_client: Anthropic):
        self.sqlite_server = sqlite_server
        self.anthropic = anthropic_client
        
    async def setup_data_tables(self) -> None:
        """Setup tables for storing extracted data."""
        try:
            
            await self.sqlite_server.execute_tool("write_query", {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL,
                    output_tokens REAL,
                    currency TEXT DEFAULT 'USD',
                    billing_period TEXT,  -- 'monthly', 'yearly', 'one-time'
                    features TEXT,  -- JSON array
                    limitations TEXT,
                    source_query TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            })
            
            logging.info("✓ Data extraction tables initialized")
            
        except Exception as e:
            logging.error(f"Failed to setup data tables: {e}")

    async def _get_structured_extraction(self, prompt: str) -> str:
        """Use Claude to extract structured data."""
        try:
            response = self.anthropic.messages.create(
                max_tokens=1024,
                model='claude-sonnet-4-5-20250929',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            text_content = ""
            for content in response.content:
                if content.type == 'text':
                    text_content += content.text
            
            return text_content.strip()
            
        except Exception as e:
            logging.error(f"Error in structured extraction: {e}")
            return '{"error": "extraction failed"}'
    
    async def extract_and_store_data(self, user_query: str, llm_response: str, 
                                   source_url: str = None) -> None:
        """Extract structured data from LLM response and store it."""
        try:            
            extraction_prompt = f"""
            Analyze this text and extract pricing information in JSON format:
            
            Text: {llm_response}
            
            Extract pricing plans with this structure:
            {{
                "company_name": "company name",
                "plans": [
                    {{
                        "plan_name": "plan name",
                        "input_tokens": number or null,
                        "output_tokens": number or null,
                        "currency": "USD",
                        "billing_period": "monthly/yearly/one-time",
                        "features": ["feature1", "feature2"],
                        "limitations": "any limitations mentioned",
                        "query": "the user's query"
                    }}
                ]
            }}
            
            Return only valid JSON, no other text. Do not return your response enclosed in ```json```
            """
            
            extraction_response = await self._get_structured_extraction(extraction_prompt)
            extraction_response = extraction_response.replace("```json\n", "").replace("```", "")
            pricing_data = json.loads(extraction_response)
            
            for plan in pricing_data.get("plans", []):
                # Format values for SQL
                values = [
                    pricing_data.get("company_name"),
                    plan.get("plan_name"),
                    plan.get("input_tokens"),
                    plan.get("output_tokens"),
                    plan.get("currency", "USD"),
                    plan.get("billing_period"),
                    json.dumps(plan.get("features", [])),
                    plan.get("limitations"),
                    user_query
                ]
                
                query = """
                    INSERT INTO pricing_plans 
                    (company_name, plan_name, input_tokens, output_tokens, currency, 
                     billing_period, features, limitations, source_query)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                await self.sqlite_server.execute_tool("write_query", {
                    "query": query,
                    "parameters": values
                })
            
            logger.info(f"Stored {len(pricing_data.get('plans', []))} pricing plans")
            
        except Exception as e:
            logging.error(f"Error extracting pricing data: {e}")


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], api_key: str) -> None:
        self.servers: list[Server] = servers
        self.anthropic = Anthropic(
            api_key=api_key,
            base_url="https://claude.vocareum.com"
        )
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_server: Dict[str, str] = {}
        self.sqlite_server: Server | None = None
        self.data_extractor: DataExtractor | None = None

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_query(self, query: str) -> None:
        """Process a user query and extract/store relevant data."""
        messages = [{'role': 'user', 'content': query}]
        model_name = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20240620')
        
        tools_to_pass = self.available_tools
        system_prompt = None
        if "compare" in query.lower():
            # Exclude scraping and text file tools to force the LLM to read from sqlite DB
            tools_to_pass = [t for t in self.available_tools if t["name"] not in ("scrape_websites", "extract_scraped_info", "read_text_file")]
            system_prompt = "You are comparing pricing data. Since this is a comparison request, you must retrieve previously stored pricing plans from the database using read_query. Scraping is disabled for comparison requests."
            print("\n[Enforcing database read for comparison query]")

        kwargs = {
            "max_tokens": 2024,
            "model": model_name,
            "tools": tools_to_pass,
            "messages": messages
        }
        if system_prompt:
            kwargs["system"] = system_prompt
            
        response = self.anthropic.messages.create(**kwargs)
        
        full_response = ""
        source_url = None
        
        while True:
            has_tool_use = any(content.type == 'tool_use' for content in response.content)
            
            for content in response.content:
                if content.type == 'text':
                    full_response += content.text
                    print(content.text, end="", flush=True)
            
            if not has_tool_use:
                messages.append({'role': 'assistant', 'content': response.content})
                break
                
            messages.append({'role': 'assistant', 'content': response.content})
            
            tool_results = []
            for content in response.content:
                if content.type == 'tool_use':
                    tool_name = content.name
                    tool_args = content.input
                    print(f"\n[Using tool: {tool_name}]")
                    
                    if tool_name in self.tool_to_server:
                        server_name = self.tool_to_server[tool_name]
                        server = next(s for s in self.servers if s.name == server_name)
                        
                        try:
                            result = await server.execute_tool(tool_name, tool_args)
                            
                            if hasattr(result, 'content'):
                                text_content = ""
                                for item in result.content:
                                    if item.type == 'text':
                                        text_content += item.text
                                        url = self._extract_url_from_result(item.text)
                                        if url:
                                            source_url = url
                                
                                tool_results.append({
                                    'type': 'tool_result',
                                    'tool_use_id': content.id,
                                    'content': text_content
                                })
                            else:
                                tool_results.append({
                                    'type': 'tool_result',
                                    'tool_use_id': content.id,
                                    'content': str(result)
                                })
                                
                        except Exception as e:
                            print(f"\nError executing tool {tool_name}: {e}")
                            tool_results.append({
                                'type': 'tool_result',
                                'tool_use_id': content.id,
                                'content': f"Error: {str(e)}",
                                'is_error': True
                            })
                    else:
                        print(f"\nUnknown tool: {tool_name}")
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': content.id,
                            'content': f"Error: Unknown tool {tool_name}",
                            'is_error': True
                        })
            
            messages.append({'role': 'user', 'content': tool_results})
            
            kwargs = {
                "max_tokens": 2024,
                "model": model_name,
                "tools": tools_to_pass,
                "messages": messages
            }
            if system_prompt:
                kwargs["system"] = system_prompt
                
            response = self.anthropic.messages.create(**kwargs)
            
        if self.data_extractor and full_response.strip():
            await self.data_extractor.extract_and_store_data(query, full_response.strip(), source_url)

    def _extract_url_from_result(self, result_text: str) -> str | None:
        """Extract URL from tool result."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, result_text)
        return urls[0] if urls else None

    async def chat_loop(self) -> None:
        """Run an interactive chat loop."""
        print("\nMCP Chatbot with Data Extraction Started!")
        print("Type your queries, 'show data' to view stored data, or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'show data':
                    await self.show_stored_data()
                    continue
                    
                await self.process_query(query)
                print("\n")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def show_stored_data(self) -> None:
        """Show recently stored data."""
        if not self.sqlite_server:
            logger.info("No database available")
            return
            
        try:
            result = await self.sqlite_server.execute_tool("read_query", {
                "query": "SELECT company_name, plan_name, input_tokens, output_tokens, currency FROM pricing_plans ORDER BY id DESC LIMIT 5"
            })
            
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if item.type == 'text':
                        data = json.loads(item.text)
                        if not data:
                            print("No data found.")
                            return
                            
                        print("\n==================================================")
                        print("Stored Pricing Plans:")
                        print("==================================================")
                        for row in data:
                            company = row.get("company_name", "Unknown")
                            plan = row.get("plan_name", "Unknown")
                            input_price = row.get("input_tokens", "N/A")
                            output_price = row.get("output_tokens", "N/A")
                            currency = row.get("currency", "USD")
                            print(f"- Company: {company}, Plan: {plan}, Input: ${input_price} {currency}/1M tokens, Output: ${output_price} {currency}/1M tokens")
                        print("==================================================\n")
            else:
                 print("No data returned or unexpected format.")
        except Exception as e:
            print(f"Error showing data: {e}")

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    if "sqlite" in server.name.lower():
                        self.sqlite_server = server
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            for server in self.servers:
                tools = await server.list_tools()
                self.available_tools.extend(tools)
                for tool in tools:
                    self.tool_to_server[tool["name"]] = server.name

            print(f"\nConnected to {len(self.servers)} server(s)")
            print(f"Available tools: {[tool['name'] for tool in self.available_tools]}")
            
            if self.sqlite_server:
                self.data_extractor = DataExtractor(self.sqlite_server, self.anthropic)
                await self.data_extractor.setup_data_tables()
                print("Data extraction enabled")

            await self.chat_loop()

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    
    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"
    
    server_config = config.load_config(config_file)
    
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())
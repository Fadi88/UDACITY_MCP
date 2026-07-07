import asyncio
from pathlib import Path
from starter_client import Configuration, Server, ChatSession

async def run_automated_evaluation():
    config = Configuration()
    script_dir = Path(__file__).parent
    config_file = script_dir / "server_config.json"
    server_config = config.load_config(config_file)
    
    servers = [Server(name, srv_config) for name, srv_config in server_config["mcpServers"].items()]
    chat_session = ChatSession(servers, config.anthropic_api_key)
    
    try:
        # Initialize servers
        for server in servers:
            await server.initialize()
            if "sqlite" in server.name.lower():
                chat_session.sqlite_server = server
                
        # List tools
        for server in servers:
            tools = await server.list_tools()
            chat_session.available_tools.extend(tools)
            for tool in tools:
                chat_session.tool_to_server[tool["name"]] = server.name
                
        if chat_session.sqlite_server:
            from starter_client import DataExtractor
            chat_session.data_extractor = DataExtractor(chat_session.sqlite_server, chat_session.anthropic)
            await chat_session.data_extractor.setup_data_tables()
            
        print("\n=== STARTING AUTOMATED QUERIES ===")
        
        # Query 1
        q1 = "How much does cloudrift ai (https://www.cloudrift.ai/inference) charge for deepseek v3?"
        print(f"\nQuery 1: {q1}\n")
        await chat_session.process_query(q1)
        
        # Query 2
        q2 = "How much does deepinfra (https://deepinfra.com/pricing) charge for deepseek v3"
        print(f"\nQuery 2: {q2}\n")
        await chat_session.process_query(q2)
        
        # Query 3
        q3 = "Compare cloudrift ai and deepinfra's costs for deepseek v3"
        print(f"\nQuery 3: {q3}\n")
        await chat_session.process_query(q3)
        
        # Show stored data
        print("\n=== SHOWING STORED DATA ===")
        await chat_session.show_stored_data()
        
    finally:
        await chat_session.cleanup_servers()

if __name__ == "__main__":
    asyncio.run(run_automated_evaluation())

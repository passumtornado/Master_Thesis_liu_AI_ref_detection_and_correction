import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary
 

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__),"..", "server", "mcp.json"))
    url ="https://ollama.com"
    # Create LLM
    
    llm = ChatOllama(
    model="qwen3-coder:480b-cloud",
    base_url=url,
    client_kwargs={
        "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
    }
)

    # Create agent with restricted tools
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        use_server_manager=True,
        disallowed_tools=["file_system", "network"]  # Restrict potentially dangerous tools
    )
    file_path = "bibtex/bibtex_files/references.bib"
    # Run the query
    result = await agent.run(
        f"show all entries from {file_path}",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
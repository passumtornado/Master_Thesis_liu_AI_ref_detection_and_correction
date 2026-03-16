#!/usr/bin/env python3
"""Quick test to see if DBLP tools work correctly."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    # Load config
    with open("server/mcp.json", "r") as f:
        config = json.load(f)
    
    mcp_servers_config = config.get("mcpServers", config)
    print(f"Config mcpServers keys: {list(mcp_servers_config.keys())}")
    
    # Connect to DBLP
    client = MultiServerMCPClient(mcp_servers_config)
    tools_list = await client.get_tools(server_name="mcp-dblp")
    
    print(f"\nTools found from mcp-dblp ({len(tools_list)} total):")
    for tool in tools_list:
        print(f"  - {tool.name}")
    
    # Try a simple search
    tools_dict = {t.name: t for t in tools_list}
    fuzzy_tool = tools_dict.get("fuzzy_title_search")
    
    if not fuzzy_tool:
        print("\n❌ fuzzy_title_search tool NOT FOUND")
        return
    
    print(f"\n✓ fuzzy_title_search tool found")
    print(f"  Tool type: {type(fuzzy_tool)}")
    print(f"  Tool description:\n{fuzzy_tool.description if fuzzy_tool.description else 'N/A'}")
    
    # Check tool schema
    if hasattr(fuzzy_tool, 'args_schema'):
        print(f"\n  Tool schema: {fuzzy_tool.args_schema}")
    
    # Try invoking it
    print("\n🔍 Attempting fuzzy_title_search for 'Advanced SMT techniques'...")
    result = await fuzzy_tool.ainvoke({
        "title": "Advanced SMT techniques for weighted model integration",
        "similarity_threshold": 0.20,
        "max_results": 5,
        "include_bibtex": False,
    })
    
    print(f"\n📦 Raw result type: {type(result).__name__}")
    print(f"📦 Raw result length: {len(result) if isinstance(result, (list, dict)) else 'N/A'}")
    
    if isinstance(result, list) and len(result) > 0:
        print(f"📦 First element type: {type(result[0]).__name__}")
        print(f"📦 First element: {result[0]}")
        if isinstance(result[0], dict) and "text" in result[0]:
            print(f"\n✓ Found 'text' field in result[0]")
            text_content = result[0]["text"]
            print(f"📦 Text content type: {type(text_content).__name__}")
            print(f"📦 Text content preview: {str(text_content)[:200]}")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(text_content)
                print(f"✓ Successfully parsed as JSON")
                print(f"  Keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
                if isinstance(parsed, dict):
                    results = parsed.get("results", [])
                    print(f"  Results count: {len(results)}")
                    if results:
                        print(f"  First result: {results[0]}")
            except Exception as e:
                print(f"❌ Failed to parse as JSON: {e}")
    elif isinstance(result, str):
        print(f"📦 Result is string: {result[:200]}")
    elif isinstance(result, dict):
        print(f"📦 Result keys: {list(result.keys())}")
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())

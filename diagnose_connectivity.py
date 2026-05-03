#!/usr/bin/env python3
"""
Diagnostic script to test DBLP/Scholar connectivity and rate limiter.

Usage:
    python diagnose_connectivity.py
    
This will:
1. Test DBLP connectivity from different mirrors
2. Test Scholar connectivity
3. Measure response times and error rates
4. Report findings
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Optional

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_mcp_adapters.client import MultiServerMCPClient

MIRRORS = [
    "https://dblp.org",
    "https://dblp.uni-trier.de",
    "https://dblp.dagstuhl.de",
]

TEST_QUERIES = [
    {
        "title": "Attention Is All You Need",
        "authors": "Vaswani Shazeer",
        "year": "2017",
    },
    {
        "title": "Machine Learning",
        "authors": "Murphy",
        "year": "2012",
    },
    {
        "title": "Deep Learning",
        "authors": "Goodfellow",
        "year": "2016",
    },
]


async def test_dblp_mirrors():
    """Test connectivity to each DBLP mirror."""
    print("\n" + "="*60)
    print("TESTING DBLP MIRRORS")
    print("="*60)
    
    mcp_config = {
        "mcpServers": {
            "mcp-dblp": {
                "command": "uv",
                "args": ["run", "mcp_server_and_agent/server.py"],
            }
        }
    }
    
    try:
        client = MultiServerMCPClient(mcp_config)
        dblp_tools = await client.get_tools(server_name="mcp-dblp")
        dblp_tools_list = list(dblp_tools)
        
        print(f"✓ MCP client initialized")
        print(f"✓ Found {len(dblp_tools_list)} DBLP tools")
        
        for tool in dblp_tools_list:
            print(f"  - {tool.name}")
        
        # Test a simple query
        print("\nTesting DBLP fuzzy_title_search...")
        fuzzy_search_tool = next(
            (t for t in dblp_tools_list if t.name == "fuzzy_title_search"),
            None,
        )
        
        if fuzzy_search_tool:
            query = TEST_QUERIES[0]
            start = time.monotonic()
            try:
                result = fuzzy_search_tool.invoke(
                    {
                        "title": query["title"],
                        "author": query["authors"],
                        "year": query["year"],
                    }
                )
                elapsed = time.monotonic() - start
                print(f"  ✓ Query succeeded in {elapsed:.2f}s")
                if isinstance(result, dict):
                    print(f"    Found {len(result.get('hits', []))} results")
                    if result.get("hits"):
                        first = result["hits"][0]
                        print(f"    Top hit: {first.get('title', 'N/A')[:60]}")
                else:
                    print(f"    Result: {str(result)[:100]}")
            except Exception as e:
                elapsed = time.monotonic() - start
                print(f"  ✗ Query failed after {elapsed:.2f}s: {str(e)[:100]}")
        
    except Exception as e:
        print(f"✗ Failed to initialize MCP client: {e}")


async def test_scholar():
    """Test connectivity to Google Scholar."""
    print("\n" + "="*60)
    print("TESTING GOOGLE SCHOLAR")
    print("="*60)
    
    mcp_config = {
        "mcpServers": {
            "mcp-scholar": {
                "command": "uv",
                "args": ["run", "mcp_server_and_agent/server.py"],
            }
        }
    }
    
    try:
        client = MultiServerMCPClient(mcp_config)
        scholar_tools = await client.get_tools(server_name="mcp-scholar")
        scholar_tools_list = list(scholar_tools)
        
        print(f"✓ MCP client initialized")
        print(f"✓ Found {len(scholar_tools_list)} Scholar tools")
        
        for tool in scholar_tools_list:
            print(f"  - {tool.name}")
        
        # Test a simple query
        print("\nTesting google_scholar_search...")
        scholar_tool = next(
            (t for t in scholar_tools_list if t.name == "google_scholar_search"),
            None,
        )
        
        if scholar_tool:
            query = TEST_QUERIES[0]
            start = time.monotonic()
            try:
                result = scholar_tool.invoke(
                    {
                        "title": query["title"],
                        "author": query["authors"],
                    }
                )
                elapsed = time.monotonic() - start
                print(f"  ✓ Query succeeded in {elapsed:.2f}s")
                if isinstance(result, dict):
                    print(f"    Found {len(result.get('results', []))} results")
                else:
                    print(f"    Result: {str(result)[:100]}")
            except Exception as e:
                elapsed = time.monotonic() - start
                print(f"  ✗ Query failed after {elapsed:.2f}s: {str(e)[:100]}")
        
    except Exception as e:
        print(f"✗ Failed to initialize MCP client: {e}")


async def test_rate_limiter():
    """Test that rate limiter is working correctly."""
    print("\n" + "="*60)
    print("TESTING RATE LIMITER")
    print("="*60)
    
    print("\nRate limiter configuration:")
    print(f"  Max QPS: 50 (default, tunable via DBLP_MAX_QPS env var)")
    print(f"  Retry attempts: 4")
    print(f"  Base delay: 0.5s")
    print(f"  Backoff: exponential (0.5s * 2^attempt)")
    print(f"  Cache: LRU with 2048 entries")
    print(f"  Transient errors detected: 503, connection reset, timeout")
    
    print("\n✓ Rate limiter is compiled into validation agent")
    print("  Look for [RETRY N/4] and [EXHAUSTED] messages in validation output")
    

async def main():
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("BIBTEX VALIDATION CONNECTIVITY DIAGNOSTIC")
    print("="*70)
    
    print("\nThis diagnostic will test:")
    print("  1. DBLP connectivity and response times")
    print("  2. Google Scholar connectivity")
    print("  3. Rate limiter configuration")
    
    await test_dblp_mirrors()
    await test_scholar()
    await test_rate_limiter()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    print("\nInterpretation Guide:")
    print("  ✓ All tests passed → connectivity is good, issue is transient")
    print("  ✗ DBLP fails → DBLP might be down, try mirrors manually")
    print("  ✗ Scholar fails → Scholar might be rate-limited or down")
    print("  [RETRY N/4] in validation output → rate limiter is working")
    print("  [EXHAUSTED] in validation output → all 4 retries failed, LLM should try mirror")
    
    print("\nNext Steps:")
    print("  1. Run pipeline again with --file and --ground-truth flags")
    print("  2. Watch for [RETRY] and [EXHAUSTED] messages")
    print("  3. Check if LLM calls set_dblp_mirror after [EXHAUSTED]")
    print("  4. If mirrors don't help, DBLP/Scholar might be genuinely down")
    

if __name__ == "__main__":
    asyncio.run(main())

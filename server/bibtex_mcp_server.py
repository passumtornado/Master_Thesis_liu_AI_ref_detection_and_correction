import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_bibtexfile, download_bibtexfile
from fastmcp import FastMCP

# Init FastMCP server
mcp = FastMCP("bibtex_mcp")


@mcp.tool()
def parse_bibtex(file_path: Optional[str] = None, directory: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """Parse BibTeX entries from file, directory, or URL
    
    Args:
        file_path: Path to a single .bib file
        url: URL to fetch BibTeX from
    
    Returns:
        Parsed entries in JSON format
    """
    entries = []
    
    try:
        if file_path:
            # Convert to absolute path if relative
            if not os.path.isabs(file_path):
                file_path = os.path.join(Path(__file__).parent.parent, file_path)
            
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}", "total_entries": 0, "entries": []}
            
            data = process_bibtexfile(file_path)
            if data is not None:
                entries.extend(data)
        elif url:
            # Download the BibTeX file from URL
            local_path = download_bibtexfile(url)
            if local_path is None:
                return {"error": f"Failed to download BibTeX file from {url}", "total_entries": 0, "entries": []}
            
            # Parse the downloaded file and export to JSON
            data = process_bibtexfile(local_path)
            if data is not None:
                entries.extend(data)
    except Exception as e:
        return {"error": str(e), "total_entries": 0, "entries": []}
    
    return {
        "total_entries": len(entries),
        "entries": entries,
    }


if __name__ == "__main__":
    mcp.run(transport='stdio')
# Add this as a static method to each agent, or put in a shared utils.py

def _extract_text(response) -> str:
    """Normalise LLM response content across different backends."""
    content = response.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return content.strip()
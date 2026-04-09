import asyncio


def _is_transient_transport_error(exc: Exception) -> bool:
    """Best-effort detection of retryable transport failures from model backends."""
    text = str(exc).lower()
    retry_markers = (
        "incomplete chunked read",
        "peer closed connection",
        "remoteprotocolerror",
        "connection reset",
        "connection aborted",
        "read timeout",
        "timed out",
        "temporarily unavailable",
        "503",
        "502",
        "429",
    )
    return any(marker in text for marker in retry_markers)


async def _ainvoke_with_retry(llm, messages, attempts: int = 3, base_delay: float = 1.0):
    """Invoke an async LLM call with bounded exponential backoff for transient errors."""
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - backend exceptions vary by provider
            last_exc = exc
            if attempt >= attempts or not _is_transient_transport_error(exc):
                raise

            wait_s = base_delay * (2 ** (attempt - 1))
            print(
                f"  [LLM retry] transient transport error on attempt {attempt}/{attempts}: {exc}. "
                f"Retrying in {wait_s:.1f}s..."
            )
            await asyncio.sleep(wait_s)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM invocation failed without a captured exception")

def _extract_text(response) -> str:
    """Normalise LLM response content across different backends."""
    content = response.content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return content.strip()
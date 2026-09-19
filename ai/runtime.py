"""Runtime configuration shared by the paper-enhancement job."""

import sys
from urllib.parse import urlparse

# A dead key or a broken provider config fails every paper, which must stop the
# workflow; an isolated gateway hiccup should not discard the whole batch.
MAX_FAILURE_RATIO = 0.02

# These providers enable thinking mode by default, and thinking mode rejects a
# forced tool_choice (structured output then fails with a 400:
# "Thinking mode does not support this tool_choice").
THINKING_DISABLED_HOSTS = ("volces.com", "deepseek.com")


def _disables_thinking(hostname: str) -> bool:
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in THINKING_DISABLED_HOSTS
    )


def build_chat_openai_kwargs(model_name: str, base_url: str, api_key: str) -> dict:
    """Build provider-safe ChatOpenAI settings from workflow configuration."""
    model_name = model_name.strip()
    base_url = base_url.strip().rstrip("/")
    api_key = api_key.strip()

    missing = [
        name
        for name, value in (
            ("MODEL_NAME", model_name),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    kwargs = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }
    hostname = urlparse(base_url).hostname or ""
    if _disables_thinking(hostname):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


def raise_if_processing_failed(
    errors: list[str],
    total: int,
    max_failure_ratio: float = MAX_FAILURE_RATIO,
) -> None:
    """Stop the workflow when the AI batch degrades, tolerating isolated failures.

    Publishing a batch of placeholder summaries silently is the failure mode this
    guard exists for, so a widespread failure still aborts the run. A handful of
    transient errors should not throw away the rest of the day's papers.
    """
    if not errors:
        return

    tolerated = int(total * max_failure_ratio)
    if len(errors) > tolerated:
        detail = "\n  ".join(errors[:10])
        raise RuntimeError(
            f"{len(errors)}/{total} paper(s) failed AI enhancement "
            f"(tolerated {tolerated}):\n  {detail}"
        )

    detail = "\n  ".join(errors[:20])
    print(
        f"⚠️  {len(errors)}/{total} paper(s) failed AI enhancement and were "
        f"skipped:\n  {detail}",
        file=sys.stderr,
    )

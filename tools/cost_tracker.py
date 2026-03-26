"""
tools/cost_tracker.py — Per-run OpenAI token & cost accumulator.

Usage in any tool:
    from tools import cost_tracker
    cost_tracker.record("gpt-4o", resp.usage.prompt_tokens, resp.usage.completion_tokens, label="Cover letter")

Usage in app.py:
    cost_tracker.reset()          # start of each run
    summary = cost_tracker.get_summary()   # after run completes
"""

from __future__ import annotations
from dataclasses import dataclass, field

# gpt-4o pricing per 1M tokens (as of 2025)
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
}
_DEFAULT_PRICING = {"input": 2.50, "output": 10.00}  # fallback


@dataclass
class _CallRecord:
    label: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


# Module-level state — reset at the start of each app run
_records: list[_CallRecord] = []


def reset() -> None:
    """Clear all accumulated records. Call at the start of each run."""
    _records.clear()


def record(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    label: str = "",
) -> None:
    """Log one API call's token usage."""
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    input_cost  = prompt_tokens     / 1_000_000 * pricing["input"]
    output_cost = completion_tokens / 1_000_000 * pricing["output"]
    _records.append(_CallRecord(
        label=label or model,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
    ))


def get_summary() -> dict:
    """Return aggregated totals and per-call breakdown."""
    total_prompt     = sum(r.prompt_tokens     for r in _records)
    total_completion = sum(r.completion_tokens for r in _records)
    total_cost       = sum(r.total_cost        for r in _records)
    return {
        "calls":             list(_records),
        "total_prompt":      total_prompt,
        "total_completion":  total_completion,
        "total_tokens":      total_prompt + total_completion,
        "total_cost_usd":    total_cost,
    }

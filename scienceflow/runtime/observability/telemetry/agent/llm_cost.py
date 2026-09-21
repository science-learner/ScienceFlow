# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMPrice:
    """USD rates per 1M text tokens."""

    input_usd_per_1m: float
    cached_input_usd_per_1m: float
    output_usd_per_1m: float


_MODEL_RE = re.compile(r"(?:^|;)model=([^;]+)")


def extract_model_from_trace_detail(detail: str | None) -> str | None:
    match = _MODEL_RE.search(str(detail or ""))
    if not match:
        return None
    model = match.group(1).strip()
    return model or None


def _norm_model_name(model: str | None) -> str:
    return str(model or "").strip().lower()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_from_obj(obj: Any) -> LLMPrice | None:
    if isinstance(obj, LLMPrice):
        return obj
    if all(
        hasattr(obj, name)
        for name in (
            "input_usd_per_1m",
            "cached_input_usd_per_1m",
            "output_usd_per_1m",
        )
    ):
        obj = {
            "input_usd_per_1m": obj.input_usd_per_1m,
            "cached_input_usd_per_1m": obj.cached_input_usd_per_1m,
            "output_usd_per_1m": obj.output_usd_per_1m,
        }
    if not isinstance(obj, dict):
        return None
    def pick(*names):
        return _float_or_none(next((obj[name] for name in names if obj.get(name) is not None), None))

    input_rate = pick("input", "cache_miss_usd_per_1m", "input_usd_per_1m", "uncached_input_usd_per_1m")
    cached_rate = pick("cached_input", "cached_input_usd_per_1m", "cache_hit_usd_per_1m", "cache")
    output_rate = pick("output", "output_usd_per_1m")
    if any(rate is None or not math.isfinite(rate) or rate < 0 for rate in (input_rate, cached_rate, output_rate)):
        return None
    return LLMPrice(input_rate, cached_rate, output_rate)


def _merge_price_table(table: dict[str, LLMPrice], raw: Any) -> None:
    if not isinstance(raw, dict):
        return
    for model, obj in raw.items():
        price = _price_from_obj(obj)
        if price is not None:
            table[_norm_model_name(model)] = price


def load_price_table(config_prices: Any = None, *, include_defaults: bool = True) -> dict[str, LLMPrice]:
    """Normalize an explicit model price table; no implicit prices are added."""
    del include_defaults  # Retained for call compatibility.
    table: dict[str, LLMPrice] = {}
    _merge_price_table(table, config_prices)
    return table


def load_model_config_prices(*paths: str | Path | None) -> dict[str, LLMPrice]:
    """Load the single supported pricing source: model registry files."""

    from inquirycraft.llm import ModelRegistry

    table: dict[str, LLMPrice] = {}
    seen = set()
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser().resolve(strict=False)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        registry = ModelRegistry.from_json(path)
        incoming = load_price_table(registry.price_table())
        for model, price in incoming.items():
            previous = table.setdefault(model, price)
            if previous != price:
                raise ValueError(f"Conflicting model pricing for {model!r}")
    return table


def resolve_price(model: str | None, table: dict[str, LLMPrice] | None = None) -> LLMPrice | None:
    prices = table if table is not None else load_price_table()
    name = _norm_model_name(model)
    if name in prices:
        return prices[name]
    return None


def estimate_llm_cost_usd(
    *,
    model: str | None,
    tokens_input: float | str | None,
    tokens_output: float | str | None,
    tokens_cached: float | str | None = 0,
    price_table: dict[str, LLMPrice] | None = None,
) -> float | None:
    price = resolve_price(model, price_table)
    if price is None:
        return None
    try:
        input_tokens = max(0, int(float(tokens_input or 0)))
        output_tokens = max(0, int(float(tokens_output or 0)))
        cached_tokens = max(0, int(float(tokens_cached or 0)))
    except (TypeError, ValueError):
        return None
    cached_tokens = min(cached_tokens, input_tokens)
    uncached_tokens = max(0, input_tokens - cached_tokens)
    return (
        uncached_tokens * price.input_usd_per_1m
        + cached_tokens * price.cached_input_usd_per_1m
        + output_tokens * price.output_usd_per_1m
    ) / 1_000_000.0


def format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 100:
        return f"${value:.2f}"
    return f"${value:,.0f}"


def benchmark_cost(incoming, outgoing, cached, timestamp, *, model='', prices=None):
    """Estimate from explicit model-config prices and complete usage evidence."""
    del timestamp  # Retained for call compatibility with persisted trace readers.
    if any(value is None for value in (incoming, outgoing, cached)):
        return None
    if not all(math.isfinite(v) for v in (incoming, outgoing, cached)):
        return None
    if min(incoming, outgoing, cached) < 0 or cached > incoming:
        return None
    price = resolve_price(model, prices if prices is not None else {})
    if price is None:
        return None
    return ((incoming - cached) * price.input_usd_per_1m + cached * price.cached_input_usd_per_1m
            + outgoing * price.output_usd_per_1m) / 1_000_000

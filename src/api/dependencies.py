"""Shared application services (stores and analyzers) for the API."""

import threading
from dataclasses import dataclass

from src.agent.memory import MemoryManager, get_memory_manager
from src.evaluation.production import ProductionEvaluator, ProductionRunStore
from src.evaluation.reliability import (
    ProductionFailureAnalyzer,
    ProductionReliabilityAnalyzer,
    ReliabilityThresholds,
)
from src.observability.store import ObservabilityAggregator, ObservabilityStore
from src.utils.config import load_config


@dataclass
class Container:
    memory_manager: MemoryManager
    obs_store: ObservabilityStore            # live chat runs
    obs_aggregator: ObservabilityAggregator
    evaluator: ProductionEvaluator           # has its own store, separate from live runs
    run_store: ProductionRunStore
    failure_analyzer: ProductionFailureAnalyzer
    reliability_analyzer: ProductionReliabilityAnalyzer
    slow_node_threshold_ms: float


_container: Container | None = None
_lock = threading.Lock()


def _build_container() -> Container:
    config = load_config()
    thresholds = ReliabilityThresholds(
        healthy_max_failure_rate=config["reliability"]["healthy_max_failure_rate"],
        degraded_max_failure_rate=config["reliability"]["degraded_max_failure_rate"],
    )
    slow_tool_ms = config["observability"]["slow_tool_threshold_ms"]

    return Container(
        memory_manager=get_memory_manager(),
        obs_store=ObservabilityStore(),
        obs_aggregator=ObservabilityAggregator(),
        evaluator=ProductionEvaluator(
            store=ObservabilityStore(),
            auto_approve=config["evaluation"]["auto_approve"],
        ),
        run_store=ProductionRunStore(),
        failure_analyzer=ProductionFailureAnalyzer(
            thresholds=thresholds,
            slow_tool_threshold_ms=slow_tool_ms,
            regression_threshold=config["reliability"]["regression_threshold"],
        ),
        reliability_analyzer=ProductionReliabilityAnalyzer(thresholds, slow_tool_ms),
        slow_node_threshold_ms=config["observability"]["slow_node_threshold_ms"],
    )


def get_container() -> Container:
    global _container
    with _lock:
        if _container is None:
            _container = _build_container()
        return _container


def reset_container() -> None:
    """Drop all in-memory state (used by tests)."""
    global _container
    with _lock:
        _container = None

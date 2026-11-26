from src.observability.models import AgentObservabilityReport
from src.observability.store import ObservabilityAggregator, ObservabilityStore
from src.observability.tracing import NodeTimer, TraceAnalyzer


def _trace():
    timer = NodeTimer({}, "agent", reset=True)
    timer.add("tool_completed", details="get_order_status", duration_ms=40.0)
    trace = timer.finish("done")
    # overwrite measured durations for deterministic assertions
    trace[-1].duration_ms = 1500.0
    return trace


def test_analyzer_report():
    report = TraceAnalyzer().build_report(_trace(), threshold_ms=1000)

    assert report.node_durations == {"agent": 1500.0}
    assert report.tool_durations == {"get_order_status": 40.0}
    assert report.slow_nodes == {"agent": 1500.0}
    assert report.slowest_node == ("agent", 1500.0)
    assert report.total_duration_ms == 1500.0


def test_node_durations_are_summed_when_node_repeats():
    trace = _trace() + _trace()
    assert TraceAnalyzer().get_node_durations(trace) == {"agent": 3000.0}


def test_timer_reset_drops_previous_turn():
    previous = {"execution_trace": _trace()}
    assert len(NodeTimer(previous, "classify", reset=True).trace) == 1
    assert len(NodeTimer(previous, "classify").trace) == len(previous["execution_trace"]) + 1


def test_aggregator_and_health():
    fast = AgentObservabilityReport(
        total_duration_ms=100, node_durations={}, tool_durations={}, slow_nodes={}
    )
    slow = AgentObservabilityReport(
        total_duration_ms=3000, node_durations={}, tool_durations={},
        slow_nodes={"agent": 3000}, slowest_node=("agent", 3000),
    )
    aggregator = ObservabilityAggregator()

    assert aggregator.build_dashboard([fast], 1000).health.status == "healthy"

    dashboard = aggregator.build_dashboard([fast, slow], 1000)
    assert dashboard.summary.total_runs == 2
    assert dashboard.summary.average_execution_time_ms == 1550.0
    assert dashboard.health.status == "degraded"
    assert dashboard.health.slowest_node == "agent"


def test_store_roundtrip():
    store = ObservabilityStore()
    report = AgentObservabilityReport(
        total_duration_ms=1, node_durations={}, tool_durations={}, slow_nodes={}
    )
    store.save("t1", report)
    assert store.get("t1") is report
    assert store.get("missing") is None

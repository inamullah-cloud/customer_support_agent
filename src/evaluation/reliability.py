"""Production failure analysis, reliability metrics, health, and regression detection."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.evaluation.production import ProductionRunResult


# ============================================================
# Models
# ============================================================

class ProductionFailureReport(BaseModel):
    """Structured report of production agent failures."""

    total_runs: int
    failed_runs: int
    failure_frequency: dict[str, int]
    slow_node_frequency: dict[str, int]
    slow_tool_frequency: dict[str, int]


class ReliabilityMetrics(BaseModel):
    """Reliability metrics for production agent runs (percentages)."""

    failure_rate: float
    success_rate: float
    category_failure_rate: float
    tool_failure_rate: float
    execution_failure_rate: float
    final_answer_failure_rate: float


class DashboardHealth(BaseModel):
    """Overall health status of the production agent."""

    status: str
    failure_rate: float
    message: str


class ReliabilityRegression(BaseModel):
    """Comparison of two runs' reliability."""

    failure_rate_change: float
    success_rate_change: float
    regression_detected: bool
    message: str


class ProductionFailureDashboard(BaseModel):
    """Complete production failure analysis dashboard."""

    report: ProductionFailureReport
    reliability: ReliabilityMetrics
    insights: list[str]
    health: DashboardHealth
    regression: ReliabilityRegression


class ReliabilitySnapshot(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    failure_rate: float


class ReliabilityByComponent(BaseModel):
    category_failure_rate: float
    tool_selection_failure_rate: float
    execution_failure_rate: float
    final_answer_failure_rate: float


class ReliabilityThresholds(BaseModel):
    """Failure-rate boundaries (percent) used to derive health status."""

    healthy_max_failure_rate: float = 5
    degraded_max_failure_rate: float = 20


class ProductionLevelReliabilityReport(BaseModel):
    snapshot: ReliabilitySnapshot
    component_reliability: ReliabilityByComponent
    status: str


# ============================================================
# Failure correlation
# ============================================================

class FailureCorrelation:
    """Relate evaluation failures to observability data for one run."""

    def __init__(self, slow_tool_threshold_ms: float = 1000.0) -> None:
        self.slow_tool_threshold_ms = slow_tool_threshold_ms

    def analyze_failure(self, result: ProductionRunResult) -> dict:
        evaluation = result.evaluation
        observability = result.observability

        failure_reasons: List[str] = []
        if not evaluation.category_correct:
            failure_reasons.append("category_failure")
        if not evaluation.tools_correct:
            failure_reasons.append("tool_selection_failure")
        if not evaluation.execution_success:
            failure_reasons.append("execution_failure")
        if not evaluation.final_answer_generated:
            failure_reasons.append("final_answer_failure")

        slow_tools: Dict[str, float] = {}
        slow_nodes: Dict[str, float] = {}
        slowest_node = None

        if observability is not None:
            slow_tools = {
                tool: duration
                for tool, duration in observability.tool_durations.items()
                if duration > self.slow_tool_threshold_ms
            }
            slow_nodes = observability.slow_nodes
            slowest_node = observability.slowest_node

        return {
            "overall_success": evaluation.overall_success,
            "failure_reasons": failure_reasons,
            "slow_nodes": slow_nodes,
            "slowest_node": slowest_node,
            "slow_tools": slow_tools,
        }


class FailureCorrelationAggregator:
    """Aggregate failure correlations across production runs."""

    def __init__(self, slow_tool_threshold_ms: float = 1000.0) -> None:
        self.failure_correlation = FailureCorrelation(slow_tool_threshold_ms)

    def analyze_runs(self, results: List[ProductionRunResult]) -> Dict[str, Any]:
        failed_runs = 0
        failure_frequency: Dict[str, int] = {}
        slow_node_frequency: Dict[str, int] = {}
        slow_tool_frequency: Dict[str, int] = {}

        for result in results:
            analysis = self.failure_correlation.analyze_failure(result)

            if not analysis["overall_success"]:
                failed_runs += 1

            for reason in analysis["failure_reasons"]:
                failure_frequency[reason] = failure_frequency.get(reason, 0) + 1

            for node in analysis["slow_nodes"]:
                slow_node_frequency[node] = slow_node_frequency.get(node, 0) + 1

            for tool in analysis["slow_tools"]:
                slow_tool_frequency[tool] = slow_tool_frequency.get(tool, 0) + 1

        return {
            "total_runs": len(results),
            "failed_runs": failed_runs,
            "failure_frequency": failure_frequency,
            "slow_node_frequency": slow_node_frequency,
            "slow_tool_frequency": slow_tool_frequency,
        }


# ============================================================
# Failure analyzer
# ============================================================

class ProductionFailureAnalyzer:
    """Build failure reports, reliability metrics, health, and regression checks."""

    def __init__(
        self,
        thresholds: Optional[ReliabilityThresholds] = None,
        slow_tool_threshold_ms: float = 1000.0,
        regression_threshold: float = 2.0,
    ) -> None:
        self.thresholds = thresholds or ReliabilityThresholds()
        self.regression_threshold = regression_threshold
        self.aggregator = FailureCorrelationAggregator(slow_tool_threshold_ms)

    def build_report(self, results: list[ProductionRunResult]) -> ProductionFailureReport:
        report = self.aggregator.analyze_runs(results)
        return ProductionFailureReport(**report)

    def generate_insights(
        self,
        report: ProductionFailureReport,
        regression: ReliabilityRegression | None = None,
    ) -> list[str]:
        if report.total_runs == 0:
            return ["No production runs available for analysis."]

        if report.failed_runs == 0:
            return ["No production failures detected."]

        top_failure, failure_count = max(
            report.failure_frequency.items(), key=lambda item: item[1], default=("None", 0)
        )
        slow_node, slow_node_count = max(
            report.slow_node_frequency.items(), key=lambda item: item[1], default=("None", 0)
        )
        slow_tool, slow_tool_count = max(
            report.slow_tool_frequency.items(), key=lambda item: item[1], default=("None", 0)
        )

        insights = [f"Most frequent failure reason: {top_failure} ({failure_count} occurrences)."]

        if slow_node_count > 0:
            insights.append(f"Most frequently slow node: {slow_node} ({slow_node_count} occurrences).")

        if slow_tool_count > 0:
            insights.append(f"Most frequently slow tool: {slow_tool} ({slow_tool_count} occurrences).")

        if regression is not None:
            if regression.regression_detected:
                insights.append(
                    "Reliability regression detected: failure rate increased by "
                    f"{regression.failure_rate_change:.2f} percentage points."
                )
            else:
                insights.append("No reliability regression detected.")

        return insights

    def calculate_reliability(self, report: ProductionFailureReport) -> ReliabilityMetrics:
        total = report.total_runs

        if total == 0:
            return ReliabilityMetrics(
                failure_rate=0.0,
                success_rate=0.0,
                category_failure_rate=0.0,
                tool_failure_rate=0.0,
                execution_failure_rate=0.0,
                final_answer_failure_rate=0.0,
            )

        failure_rate = (report.failed_runs / total) * 100.0
        freq = report.failure_frequency

        def pct(key: str) -> float:
            return round((freq.get(key, 0) / total) * 100.0, 2)

        return ReliabilityMetrics(
            failure_rate=round(failure_rate, 2),
            success_rate=round(100.0 - failure_rate, 2),
            category_failure_rate=pct("category_failure"),
            tool_failure_rate=pct("tool_selection_failure"),
            execution_failure_rate=pct("execution_failure"),
            final_answer_failure_rate=pct("final_answer_failure"),
        )

    def determine_status(self, failure_rate: float) -> str:
        if failure_rate <= self.thresholds.healthy_max_failure_rate:
            return "healthy"
        if failure_rate <= self.thresholds.degraded_max_failure_rate:
            return "degraded"
        return "critical"

    def assess_health(self, reliability: ReliabilityMetrics) -> DashboardHealth:
        rate = reliability.failure_rate
        status = self.determine_status(rate)

        messages = {
            "healthy": f"The agent is performing well with a {rate}% failure rate.",
            "degraded": f"Agent health is degraded with a {rate}% failure rate.",
            "critical": f"Agent health is critical with a {rate}% failure rate.",
        }
        return DashboardHealth(status=status, failure_rate=rate, message=messages[status])

    def detect_reliability_regression(
        self,
        previous: ReliabilityMetrics,
        current: ReliabilityMetrics,
        threshold: float | None = None,
    ) -> ReliabilityRegression:
        threshold = self.regression_threshold if threshold is None else threshold
        failure_change = round(current.failure_rate - previous.failure_rate, 2)
        success_change = round(current.success_rate - previous.success_rate, 2)

        if failure_change > threshold:
            return ReliabilityRegression(
                failure_rate_change=failure_change,
                success_rate_change=success_change,
                regression_detected=True,
                message=f"Reliability dropped: failure rate increased by {failure_change} percentage points.",
            )

        return ReliabilityRegression(
            failure_rate_change=failure_change,
            success_rate_change=success_change,
            regression_detected=False,
            message=f"No regression: failure rate changed by {failure_change} percentage points.",
        )

    def build_dashboard(
        self,
        results: list[ProductionRunResult],
        previous_results: list[ProductionRunResult] | None = None,
    ) -> ProductionFailureDashboard:
        current_report = self.build_report(results)
        current_reliability = self.calculate_reliability(current_report)

        if previous_results:
            previous_reliability = self.calculate_reliability(self.build_report(previous_results))
            regression = self.detect_reliability_regression(previous_reliability, current_reliability)
        else:
            regression = ReliabilityRegression(
                failure_rate_change=0.0,
                success_rate_change=0.0,
                regression_detected=False,
                message="No previous runs available for comparison.",
            )

        return ProductionFailureDashboard(
            report=current_report,
            reliability=current_reliability,
            insights=self.generate_insights(current_report, regression if previous_results else None),
            health=self.assess_health(current_reliability),
            regression=regression,
        )


# ============================================================
# Reliability analyzer
# ============================================================

class ProductionReliabilityAnalyzer:
    """Reliability snapshots, per-component breakdowns, and status."""

    def __init__(
        self,
        thresholds: Optional[ReliabilityThresholds] = None,
        slow_tool_threshold_ms: float = 1000.0,
    ) -> None:
        self.thresholds = thresholds or ReliabilityThresholds()
        self.analyzer = ProductionFailureAnalyzer(self.thresholds, slow_tool_threshold_ms)

    def _snapshot_from_report(self, report: ProductionFailureReport) -> ReliabilitySnapshot:
        if report.total_runs == 0:
            return ReliabilitySnapshot(
                total_runs=0, successful_runs=0, failed_runs=0, success_rate=0.0, failure_rate=0.0
            )

        successful = report.total_runs - report.failed_runs
        return ReliabilitySnapshot(
            total_runs=report.total_runs,
            successful_runs=successful,
            failed_runs=report.failed_runs,
            success_rate=round((successful / report.total_runs) * 100.0, 2),
            failure_rate=round((report.failed_runs / report.total_runs) * 100.0, 2),
        )

    def _component_reliability_from_report(self, report: ProductionFailureReport) -> ReliabilityByComponent:
        reliability = self.analyzer.calculate_reliability(report)
        return ReliabilityByComponent(
            category_failure_rate=reliability.category_failure_rate,
            tool_selection_failure_rate=reliability.tool_failure_rate,
            execution_failure_rate=reliability.execution_failure_rate,
            final_answer_failure_rate=reliability.final_answer_failure_rate,
        )

    def calculate_snapshot(self, results: list[ProductionRunResult]) -> ReliabilitySnapshot:
        return self._snapshot_from_report(self.analyzer.build_report(results))

    def calculate_component_reliability(self, results: list[ProductionRunResult]) -> ReliabilityByComponent:
        return self._component_reliability_from_report(self.analyzer.build_report(results))

    def determine_status(self, failure_rate: float) -> str:
        return self.analyzer.determine_status(failure_rate)

    def analyze(self, results: list[ProductionRunResult]) -> ProductionLevelReliabilityReport:
        report = self.analyzer.build_report(results)
        snapshot = self._snapshot_from_report(report)

        return ProductionLevelReliabilityReport(
            snapshot=snapshot,
            component_reliability=self._component_reliability_from_report(report),
            status=self.determine_status(snapshot.failure_rate),
        )


# ============================================================
# Console report
# ============================================================

def print_reliability_dashboard(dashboard: ProductionFailureDashboard, average_latency_ms: float) -> None:
    """Print the production reliability dashboard."""
    r = dashboard.reliability
    print("=" * 40)
    print("      PRODUCTION RELIABILITY REPORT")
    print("=" * 40 + "\n")
    print(f"Status:               {dashboard.health.status.upper()}")
    print(f"Message:              {dashboard.health.message}")
    print(f"Total Runs:           {dashboard.report.total_runs}")
    print(f"Failed Runs:          {dashboard.report.failed_runs}")
    print(f"Success Rate:         {r.success_rate:>5.1f}%")
    print(f"Failure Rate:         {r.failure_rate:>5.1f}%")
    print(f"Average Latency:      {average_latency_ms:.0f} ms\n")
    print("Failure rate by component:")
    print(f"  Category:           {r.category_failure_rate:>5.1f}%")
    print(f"  Tool selection:     {r.tool_failure_rate:>5.1f}%")
    print(f"  Execution:          {r.execution_failure_rate:>5.1f}%")
    print(f"  Final answer:       {r.final_answer_failure_rate:>5.1f}%\n")
    print("Insights:")
    for insight in dashboard.insights:
        print(f"  - {insight}")
    print()

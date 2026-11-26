from src.evaluation.metrics import EvaluationResult
from src.evaluation.production import ProductionRunResult
from src.evaluation.reliability import (
    ProductionFailureAnalyzer,
    ProductionReliabilityAnalyzer,
    ReliabilityThresholds,
)


def _run(success: bool, reason_flags=None) -> ProductionRunResult:
    flags = {"category_correct": True, "tools_correct": True,
             "execution_success": True, "final_answer_generated": True}
    flags.update(reason_flags or {})
    return ProductionRunResult(
        evaluation=EvaluationResult(
            query="q", expected_category="a", predicted_category="a",
            expected_tools=["t"], predicted_tools=["t"],
            overall_success=success, **flags,
        )
    )


def test_failure_report_and_metrics():
    results = [_run(True), _run(True), _run(False, {"tools_correct": False}), _run(True)]
    analyzer = ProductionFailureAnalyzer()

    report = analyzer.build_report(results)
    assert (report.total_runs, report.failed_runs) == (4, 1)
    assert report.failure_frequency == {"tool_selection_failure": 1}

    reliability = analyzer.calculate_reliability(report)
    assert reliability.failure_rate == 25.0
    assert reliability.tool_failure_rate == 25.0
    assert analyzer.assess_health(reliability).status == "critical"


def test_health_thresholds_have_no_gap():
    analyzer = ProductionFailureAnalyzer(ReliabilityThresholds())
    assert analyzer.determine_status(3) == "healthy"
    assert analyzer.determine_status(10) == "degraded"
    assert analyzer.determine_status(50) == "critical"


def test_regression_detection():
    analyzer = ProductionFailureAnalyzer()
    good = [_run(True)] * 4
    bad = [_run(True)] * 2 + [_run(False, {"execution_success": False})] * 2

    dashboard = analyzer.build_dashboard(bad, previous_results=good)
    assert dashboard.regression.regression_detected is True
    assert dashboard.regression.failure_rate_change == 50.0

    first_run = analyzer.build_dashboard(bad, previous_results=[])
    assert first_run.regression.regression_detected is False  # nothing to compare against


def test_reliability_analyzer():
    report = ProductionReliabilityAnalyzer(ReliabilityThresholds()).analyze(
        [_run(True), _run(False, {"category_correct": False})]
    )
    assert report.snapshot.success_rate == 50.0
    assert report.component_reliability.category_failure_rate == 50.0
    assert report.status == "critical"

"""Entry point: run the evaluation dataset and print evaluation + reliability reports."""

from src.agent.graph import app
from src.evaluation.metrics import calculate_metrics, print_evaluation_report
from src.evaluation.production import ProductionEvaluator, ProductionRunStore
from src.evaluation.reliability import (
    ProductionFailureAnalyzer,
    ReliabilityThresholds,
    print_reliability_dashboard,
)
from src.evaluation.test_cases import EVAL_DATASET
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main() -> None:
    config = load_config()
    setup_logging(config["logging"]["level"])

    evaluator = ProductionEvaluator(auto_approve=config["evaluation"]["auto_approve"])
    run_store = ProductionRunStore()

    results = [evaluator.evaluate_run(case, app) for case in EVAL_DATASET]
    run_store.save_batch(results)

    evaluations = [r.evaluation for r in results]
    print_evaluation_report(evaluations, calculate_metrics(evaluations))

    analyzer = ProductionFailureAnalyzer(
        thresholds=ReliabilityThresholds(
            healthy_max_failure_rate=config["reliability"]["healthy_max_failure_rate"],
            degraded_max_failure_rate=config["reliability"]["degraded_max_failure_rate"],
        ),
        slow_tool_threshold_ms=config["observability"]["slow_tool_threshold_ms"],
        regression_threshold=config["reliability"]["regression_threshold"],
    )
    dashboard = analyzer.build_dashboard(results, previous_results=[])
    print_reliability_dashboard(dashboard, run_store.get_average_latency(results))


if __name__ == "__main__":
    main()

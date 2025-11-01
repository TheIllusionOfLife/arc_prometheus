"""
Tests for scripts/analyze_baseline.py - Baseline validation analysis script.
"""

import json

import pytest


class TestAnalyzeBaseline:
    """Test suite for analyze_baseline.py functionality."""

    @pytest.fixture
    def mock_summary_high_score(self, tmp_path):
        """Create mock summary.json with high score (≥8%)."""
        exp_dir = tmp_path / "high_score"
        exp_dir.mkdir()
        summary = {
            "total_tasks": 40,
            "successful_tasks": 35,
            "failed_tasks": 5,
            "success_rate": 87.5,
            "avg_final_fitness": 5.2,
            "median_final_fitness": 4.0,
            "avg_generations": 2.8,
            "avg_time_per_task": 45.3,
            "total_time": 1812.0,
            "test_accuracy_pct": 10.5,  # High score
            "train_accuracy_pct": 65.0,
            "tasks_with_positive_fitness": 32,
            "perfect_solvers": 2,
            "error_distribution": {
                "timeout": 3,
                "syntax_error": 2,
            },
        }
        summary_file = exp_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)
        return exp_dir

    @pytest.fixture
    def mock_summary_medium_score(self, tmp_path):
        """Create mock summary.json with medium score (5-8%)."""
        exp_dir = tmp_path / "medium_score"
        exp_dir.mkdir()
        summary = {
            "total_tasks": 40,
            "successful_tasks": 28,
            "failed_tasks": 12,
            "success_rate": 70.0,
            "avg_final_fitness": 2.8,
            "median_final_fitness": 2.0,
            "avg_generations": 2.5,
            "avg_time_per_task": 42.0,
            "total_time": 1680.0,
            "test_accuracy_pct": 6.5,  # Medium score
            "train_accuracy_pct": 55.0,
            "tasks_with_positive_fitness": 25,
            "perfect_solvers": 0,
            "error_distribution": {
                "timeout": 8,
                "syntax_error": 4,
            },
        }
        summary_file = exp_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)
        return exp_dir

    @pytest.fixture
    def mock_summary_low_score(self, tmp_path):
        """Create mock summary.json with low score (<5%)."""
        exp_dir = tmp_path / "low_score"
        exp_dir.mkdir()
        summary = {
            "total_tasks": 40,
            "successful_tasks": 18,
            "failed_tasks": 22,
            "success_rate": 45.0,
            "avg_final_fitness": 1.2,
            "median_final_fitness": 0.0,
            "avg_generations": 2.2,
            "avg_time_per_task": 38.5,
            "total_time": 1540.0,
            "test_accuracy_pct": 3.2,  # Low score
            "train_accuracy_pct": 42.0,
            "tasks_with_positive_fitness": 15,
            "perfect_solvers": 0,
            "error_distribution": {
                "timeout": 15,
                "syntax_error": 7,
            },
        }
        summary_file = exp_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)
        return exp_dir

    @pytest.fixture
    def mock_quick_test_summary(self, tmp_path):
        """Create mock summary.json for quick test (10 tasks)."""
        exp_dir = tmp_path / "quick_test"
        exp_dir.mkdir()
        summary = {
            "total_tasks": 10,
            "successful_tasks": 8,
            "failed_tasks": 2,
            "success_rate": 80.0,
            "avg_final_fitness": 3.5,
            "median_final_fitness": 3.0,
            "avg_generations": 2.5,
            "avg_time_per_task": 38.2,  # Used for sample size calculation
            "total_time": 382.0,
            "test_accuracy_pct": 8.5,
            "train_accuracy_pct": 60.0,
            "tasks_with_positive_fitness": 7,
            "perfect_solvers": 1,
            "error_distribution": {
                "timeout": 1,
                "syntax_error": 1,
            },
        }
        summary_file = exp_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)
        return exp_dir

    def test_load_summary_success(self, mock_summary_high_score):
        """Test loading summary.json from directory."""
        from scripts.analyze_baseline import load_summary

        summary = load_summary(str(mock_summary_high_score))
        assert summary["total_tasks"] == 40
        assert summary["test_accuracy_pct"] == 10.5

    def test_load_summary_missing_file(self, tmp_path):
        """Test loading from directory without summary.json."""
        from scripts.analyze_baseline import load_summary

        with pytest.raises(FileNotFoundError):
            load_summary(str(tmp_path))

    def test_calculate_sample_size_fast_tasks(self, mock_quick_test_summary):
        """Test sample size calculation for fast tasks (<1 min each)."""
        from scripts.analyze_baseline import calculate_sample_size, load_summary

        summary = load_summary(str(mock_quick_test_summary))
        sample_size, estimated_time = calculate_sample_size(
            summary, target_duration=3600
        )

        # 3600s / 38.2s = 94 tasks, capped at 50
        assert sample_size == 50
        assert estimated_time == pytest.approx(50 * 38.2, rel=0.01)

    def test_calculate_sample_size_slow_tasks(self, tmp_path):
        """Test sample size calculation for slow tasks (>1 min each)."""
        from scripts.analyze_baseline import calculate_sample_size, load_summary

        # Create summary with slow tasks
        summary = {
            "total_tasks": 10,
            "avg_time_per_task": 120.0,  # 2 minutes per task
            "total_time": 1200.0,
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))
        sample_size, estimated_time = calculate_sample_size(
            loaded, target_duration=3600
        )

        # 3600s / 120s = 30 tasks
        assert sample_size == 30
        assert estimated_time == pytest.approx(30 * 120.0, rel=0.01)

    def test_calculate_sample_size_custom_duration(self, mock_quick_test_summary):
        """Test sample size calculation with custom target duration."""
        from scripts.analyze_baseline import calculate_sample_size, load_summary

        summary = load_summary(str(mock_quick_test_summary))
        sample_size, estimated_time = calculate_sample_size(
            summary,
            target_duration=1800,  # 30 minutes
        )

        # 1800s / 38.2s = 47 tasks
        assert sample_size == 47
        assert estimated_time == pytest.approx(47 * 38.2, rel=0.01)

    def test_get_decision_high_score(self, mock_summary_high_score):
        """Test decision gate for high score (≥8%)."""
        from scripts.analyze_baseline import get_decision, load_summary

        summary = load_summary(str(mock_summary_high_score))
        decision = get_decision(summary)

        assert decision["category"] == "high"
        assert decision["next_phase"] == "Phase 2a (Active Inference)"
        assert "8% baseline + 5% Active Inference" in decision["reasoning"]

    def test_get_decision_medium_score(self, mock_summary_medium_score):
        """Test decision gate for medium score (5-8%)."""
        from scripts.analyze_baseline import get_decision, load_summary

        summary = load_summary(str(mock_summary_medium_score))
        decision = get_decision(summary)

        assert decision["category"] == "medium"
        assert decision["next_phase"] == "Phase 2b (Hyperparameter Tuning)"
        assert "stronger baseline" in decision["reasoning"]

    def test_get_decision_low_score(self, mock_summary_low_score):
        """Test decision gate for low score (<5%)."""
        from scripts.analyze_baseline import get_decision, load_summary

        summary = load_summary(str(mock_summary_low_score))
        decision = get_decision(summary)

        assert decision["category"] == "low"
        assert decision["next_phase"] == "Fundamental rethink needed"
        assert "unlikely to reach competitive threshold" in decision["reasoning"]

    def test_get_decision_boundary_high(self, tmp_path):
        """Test decision gate at 8.0% boundary (exactly high)."""
        from scripts.analyze_baseline import get_decision, load_summary

        summary = {"test_accuracy_pct": 8.0}
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))
        decision = get_decision(loaded)
        assert decision["category"] == "high"

    def test_get_decision_boundary_medium(self, tmp_path):
        """Test decision gate at 5.0% boundary (exactly medium)."""
        from scripts.analyze_baseline import get_decision, load_summary

        summary = {"test_accuracy_pct": 5.0}
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))
        decision = get_decision(loaded)
        assert decision["category"] == "medium"

    def test_format_statistics_output(self, mock_summary_high_score):
        """Test formatting of statistics for display."""
        from scripts.analyze_baseline import format_statistics, load_summary

        summary = load_summary(str(mock_summary_high_score))
        output = format_statistics(summary)

        # Check key statistics are included (flexible formatting)
        assert "10.5" in output or "10.50" in output  # test accuracy
        assert "65.0" in output or "65.00" in output  # train accuracy
        assert "32/40" in output or "32" in output  # tasks solved
        assert "5.2" in output or "5.20" in output  # avg fitness

    def test_compare_experiments_improvement(
        self, mock_summary_medium_score, mock_summary_high_score
    ):
        """Test comparison showing improvement."""
        from scripts.analyze_baseline import compare_experiments, load_summary

        baseline = load_summary(str(mock_summary_medium_score))
        improved = load_summary(str(mock_summary_high_score))

        comparison = compare_experiments(baseline, improved, "Baseline", "Improved")

        # Check improvement detected
        assert comparison["test_accuracy_delta"] == pytest.approx(10.5 - 6.5, rel=0.01)
        assert comparison["test_accuracy_delta_pct"] > 0
        assert comparison["improvement"] is True

    def test_compare_experiments_regression(
        self, mock_summary_high_score, mock_summary_low_score
    ):
        """Test comparison showing regression."""
        from scripts.analyze_baseline import compare_experiments, load_summary

        baseline = load_summary(str(mock_summary_high_score))
        regressed = load_summary(str(mock_summary_low_score))

        comparison = compare_experiments(baseline, regressed, "Baseline", "Regressed")

        # Check regression detected
        assert comparison["test_accuracy_delta"] == pytest.approx(3.2 - 10.5, rel=0.01)
        assert comparison["test_accuracy_delta_pct"] < 0
        assert comparison["improvement"] is False

    def test_format_comparison_output(
        self, mock_summary_medium_score, mock_summary_high_score
    ):
        """Test formatting of comparison for display."""
        from scripts.analyze_baseline import (
            compare_experiments,
            format_comparison,
            load_summary,
        )

        baseline = load_summary(str(mock_summary_medium_score))
        improved = load_summary(str(mock_summary_high_score))
        comparison = compare_experiments(baseline, improved, "Baseline", "Improved")

        output = format_comparison(comparison)

        # Check comparison includes both values and delta
        assert "6.5" in output or "6.50" in output  # baseline test accuracy
        assert "10.5" in output or "10.50" in output  # improved test accuracy
        assert "+" in output  # positive delta indicator

    def test_cli_single_experiment(self, mock_summary_high_score, capsys):
        """Test CLI with single experiment directory."""
        # Simulate CLI args
        import sys

        from scripts.analyze_baseline import main

        sys.argv = ["analyze_baseline.py", str(mock_summary_high_score)]

        main()
        captured = capsys.readouterr()

        # Check output includes key info
        assert "10.5" in captured.out  # test accuracy
        assert "Phase 2a" in captured.out  # decision

    def test_cli_compare_mode(
        self, mock_summary_medium_score, mock_summary_high_score, capsys
    ):
        """Test CLI with two experiment directories (compare mode)."""
        import sys

        from scripts.analyze_baseline import main

        sys.argv = [
            "analyze_baseline.py",
            str(mock_summary_medium_score),
            str(mock_summary_high_score),
            "--compare",
        ]

        main()
        captured = capsys.readouterr()

        # Check comparison output
        assert "6.5" in captured.out  # baseline score
        assert "10.5" in captured.out  # improved score
        assert "Δ" in captured.out or "delta" in captured.out.lower()

    def test_cli_sample_size_mode(self, mock_quick_test_summary, capsys):
        """Test CLI with --sample-size flag."""
        import sys

        from scripts.analyze_baseline import main

        sys.argv = [
            "analyze_baseline.py",
            str(mock_quick_test_summary),
            "--sample-size",
        ]

        main()
        captured = capsys.readouterr()

        # Check sample size calculation output
        assert "SAMPLE SIZE CALCULATION" in captured.out
        assert "Sample Size:" in captured.out
        assert "Estimated Time:" in captured.out
        assert "Target Duration:" in captured.out

    def test_quick_test_analysis_workflow(self, mock_quick_test_summary, capsys):
        """Integration test: Full quick test analysis workflow."""
        from scripts.analyze_baseline import (
            calculate_sample_size,
            format_statistics,
            get_decision,
            load_summary,
        )

        # Load quick test results
        summary = load_summary(str(mock_quick_test_summary))

        # Calculate sample size
        sample_size, estimated_time = calculate_sample_size(summary)

        # Get decision (for quick test, just informational)
        decision = get_decision(summary)

        # Format output
        stats_output = format_statistics(summary)

        # Verify workflow results
        assert sample_size <= 50
        assert estimated_time > 0
        assert decision["category"] in ["high", "medium", "low"]
        assert "8.5" in stats_output  # test accuracy from quick test

    def test_missing_optional_fields(self, tmp_path):
        """Test handling of summary with missing optional fields."""
        from scripts.analyze_baseline import format_statistics, load_summary

        # Minimal summary
        summary = {
            "total_tasks": 10,
            "test_accuracy_pct": 5.0,
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))
        # Should not crash with missing fields
        output = format_statistics(loaded)
        assert "5.0" in output

    def test_zero_division_protection(self, tmp_path):
        """Test protection against zero division errors."""
        from scripts.analyze_baseline import calculate_sample_size, load_summary

        # Summary with zero time per task (edge case)
        summary = {
            "total_tasks": 10,
            "avg_time_per_task": 0.0,
            "total_time": 0.0,
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))

        # Should handle gracefully (return max sample size)
        sample_size, estimated_time = calculate_sample_size(loaded)
        assert sample_size == 50  # Max cap
        assert estimated_time == 0.0

    def test_format_statistics_zero_tasks(self, tmp_path):
        """Test format_statistics with zero total tasks to prevent division by zero."""
        from scripts.analyze_baseline import format_statistics, load_summary

        summary = {
            "total_tasks": 0,
            "tasks_with_positive_fitness": 0,
            "test_accuracy_pct": 0.0,
            "train_accuracy_pct": 0.0,
            "avg_final_fitness": 0.0,
            "median_final_fitness": 0.0,
            "avg_generations": 0.0,
            "avg_time_per_task": 0.0,
            "total_time": 0.0,
            "perfect_solvers": 0,
        }
        summary_file = tmp_path / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        loaded = load_summary(str(tmp_path))
        # This should not raise ZeroDivisionError after the fix
        output = format_statistics(loaded)
        assert "0/0" in output
        assert "0.0%" in output  # Should show 0.0% not crash

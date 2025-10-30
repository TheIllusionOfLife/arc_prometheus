"""Tests for benchmark_evolution.py - Real-world evolution loop benchmarking."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTaskSelection:
    """Test task selection and loading functionality."""

    def test_load_task_ids_from_file(self, tmp_path):
        """Test loading task IDs from a text file."""
        from scripts.benchmark_evolution import load_task_ids_from_file

        # Create task IDs file
        task_ids_file = tmp_path / "task_ids.txt"
        task_ids_file.write_text("00576224\n007bbfb7\n025d127b\n")

        task_ids = load_task_ids_from_file(str(task_ids_file))

        assert len(task_ids) == 3
        assert task_ids == ["00576224", "007bbfb7", "025d127b"]

    def test_load_task_ids_ignores_empty_lines_and_comments(self, tmp_path):
        """Test that empty lines and comments are ignored."""
        from scripts.benchmark_evolution import load_task_ids_from_file

        task_ids_file = tmp_path / "task_ids.txt"
        task_ids_file.write_text("""
# Comment line
00576224

007bbfb7  # Inline comment
# Another comment
025d127b
""")

        task_ids = load_task_ids_from_file(str(task_ids_file))

        assert len(task_ids) == 3
        assert task_ids == ["00576224", "007bbfb7", "025d127b"]

    def test_random_sample_tasks(self, tmp_path):
        """Test random sampling from training dataset."""
        from scripts.benchmark_evolution import random_sample_tasks

        # Create mock training challenges file
        training_data = {f"task_{i:08d}": {"train": [], "test": []} for i in range(100)}

        training_file = tmp_path / "training_challenges.json"
        training_file.write_text(json.dumps(training_data))

        # Sample 10 random tasks
        task_ids = random_sample_tasks(str(training_file), n=10, seed=42)

        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10  # All unique
        assert all(tid in training_data for tid in task_ids)

    def test_random_sample_deterministic_with_seed(self, tmp_path):
        """Test that random sampling is deterministic with same seed."""
        from scripts.benchmark_evolution import random_sample_tasks

        training_data = {f"task_{i:08d}": {"train": [], "test": []} for i in range(100)}
        training_file = tmp_path / "training_challenges.json"
        training_file.write_text(json.dumps(training_data))

        # Sample with same seed twice
        sample1 = random_sample_tasks(str(training_file), n=10, seed=42)
        sample2 = random_sample_tasks(str(training_file), n=10, seed=42)

        assert sample1 == sample2

    def test_parse_task_ids_from_cli(self):
        """Test parsing comma-separated task IDs from CLI."""
        from scripts.benchmark_evolution import parse_task_ids

        task_ids = parse_task_ids("00576224,007bbfb7,025d127b")

        assert len(task_ids) == 3
        assert task_ids == ["00576224", "007bbfb7", "025d127b"]

    def test_parse_task_ids_strips_whitespace(self):
        """Test that whitespace around task IDs is stripped."""
        from scripts.benchmark_evolution import parse_task_ids

        task_ids = parse_task_ids("  00576224 , 007bbfb7  ,025d127b  ")

        assert task_ids == ["00576224", "007bbfb7", "025d127b"]


class TestBenchmarkExecution:
    """Test benchmark execution and result collection."""

    @patch("scripts.benchmark_evolution.run_evolution_loop")
    def test_run_single_task_benchmark(self, mock_run_loop, tmp_path):
        """Test benchmarking a single task."""
        from scripts.benchmark_evolution import run_single_task_benchmark

        # Mock evolution loop results
        mock_run_loop.return_value = [
            {
                "generation": 0,
                "solver_code": "def solve(x): return x",
                "fitness_result": {
                    "fitness": 5.0,
                    "train_correct": 2,
                    "train_total": 3,
                    "test_correct": 0,
                    "test_total": 1,
                    "train_accuracy": 0.67,
                    "test_accuracy": 0.0,
                    "execution_errors": [],
                    "error_details": [],
                    "error_summary": {},
                },
                "refinement_count": 0,
                "total_time": 2.5,
                "improvement": 0.0,
            }
        ]

        # Create mock task file with valid train/test structure
        task_data = {
            "train": [
                {"input": [[1]], "output": [[2]]},
                {"input": [[3]], "output": [[6]]},
                {"input": [[5]], "output": [[10]]},
            ],
            "test": [{"input": [[7]], "output": [[14]]}],
        }
        training_file = tmp_path / "training.json"
        training_file.write_text(json.dumps({"00576224": task_data}))

        result = run_single_task_benchmark(
            task_id="00576224",
            training_challenges_path=str(training_file),
            max_generations=5,
            sandbox_mode="multiprocess",
            model_name="gemini-2.5-flash-lite",
            programmer_temperature=0.3,
            refiner_temperature=0.4,
            timeout_eval=5,
            timeout_llm=60,
            use_cache=True,
        )

        # Verify result structure
        assert result["task_id"] == "00576224"
        assert result["success"] is True
        assert "generations" in result
        assert len(result["generations"]) == 1
        assert result["final_fitness"] == 5.0
        assert result["total_generations"] == 1
        assert "total_time" in result
        assert "error" is None or "error" not in result

    @patch("scripts.benchmark_evolution.run_evolution_loop")
    def test_run_single_task_benchmark_handles_exceptions(
        self, mock_run_loop, tmp_path
    ):
        """Test that exceptions during benchmarking are captured."""
        from scripts.benchmark_evolution import run_single_task_benchmark

        # Mock exception
        mock_run_loop.side_effect = Exception("LLM API timeout")

        # Create valid task structure to avoid data_loader errors
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[6]]}],
        }
        training_file = tmp_path / "training.json"
        training_file.write_text(json.dumps({"00576224": task_data}))

        result = run_single_task_benchmark(
            task_id="00576224",
            training_challenges_path=str(training_file),
            max_generations=5,
            sandbox_mode="multiprocess",
        )

        assert result["task_id"] == "00576224"
        assert result["success"] is False
        assert "error" in result
        assert "LLM API timeout" in result["error"]

    def test_save_task_result_to_json(self, tmp_path):
        """Test saving individual task result to JSON file."""
        from scripts.benchmark_evolution import save_task_result

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        result = {
            "task_id": "00576224",
            "success": True,
            "final_fitness": 13.0,
            "total_generations": 3,
            "generations": [],
        }

        save_task_result(result, str(output_dir))

        # Verify file created
        result_file = output_dir / "task_00576224.json"
        assert result_file.exists()

        # Verify content
        with open(result_file) as f:
            saved = json.load(f)
        assert saved["task_id"] == "00576224"
        assert saved["final_fitness"] == 13.0


class TestMetadataGeneration:
    """Test experiment metadata generation."""

    @patch("scripts.benchmark_evolution.subprocess.run")
    def test_generate_experiment_metadata(self, mock_subprocess):
        """Test generating experiment metadata."""
        from scripts.benchmark_evolution import generate_experiment_metadata

        # Mock git commit hash
        mock_subprocess.return_value.stdout = "abc123def456"
        mock_subprocess.return_value.returncode = 0

        metadata = generate_experiment_metadata(
            experiment_name="test_run",
            task_ids=["00576224", "007bbfb7"],
            config={
                "model": "gemini-2.5-flash-lite",
                "max_generations": 5,
                "sandbox_mode": "multiprocess",
            },
        )

        assert metadata["experiment_name"] == "test_run"
        assert metadata["num_tasks"] == 2
        assert metadata["task_ids"] == ["00576224", "007bbfb7"]
        assert metadata["config"]["model"] == "gemini-2.5-flash-lite"
        assert "timestamp" in metadata
        assert "git_commit" in metadata

    def test_generate_experiment_metadata_without_git(self):
        """Test metadata generation when git is not available."""
        from scripts.benchmark_evolution import generate_experiment_metadata

        # This should not crash even if git is unavailable
        metadata = generate_experiment_metadata(
            experiment_name="test_run",
            task_ids=["00576224"],
            config={"model": "test"},
        )

        assert metadata["experiment_name"] == "test_run"
        # git_commit may be None or "unknown"
        assert "git_commit" in metadata


class TestAggregateStatistics:
    """Test aggregate statistics calculation."""

    def test_calculate_aggregate_statistics(self):
        """Test calculating aggregate statistics from task results."""
        from scripts.benchmark_evolution import calculate_aggregate_statistics

        task_results = [
            {
                "task_id": "task1",
                "success": True,
                "final_fitness": 13.0,
                "total_generations": 2,
                "total_time": 10.5,
                "generations": [
                    {"fitness_result": {"error_summary": {"syntax": 1}}},
                    {"fitness_result": {"error_summary": {}}},
                ],
            },
            {
                "task_id": "task2",
                "success": True,
                "final_fitness": 5.0,
                "total_generations": 5,
                "total_time": 25.3,
                "generations": [
                    {"fitness_result": {"error_summary": {"timeout": 2}}},
                    {"fitness_result": {"error_summary": {"logic": 1}}},
                    {"fitness_result": {"error_summary": {}}},
                    {"fitness_result": {"error_summary": {}}},
                    {"fitness_result": {"error_summary": {}}},
                ],
            },
            {
                "task_id": "task3",
                "success": False,
                "error": "LLM timeout",
            },
        ]

        stats = calculate_aggregate_statistics(task_results)

        assert stats["total_tasks"] == 3
        assert stats["successful_tasks"] == 2
        assert stats["failed_tasks"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["avg_final_fitness"] == (13.0 + 5.0) / 2
        assert stats["avg_generations"] == (2 + 5) / 2
        assert stats["avg_time_per_task"] == (10.5 + 25.3) / 2
        assert "error_distribution" in stats
        assert stats["error_distribution"]["syntax"] == 1
        assert stats["error_distribution"]["timeout"] == 2
        assert stats["error_distribution"]["logic"] == 1


class TestCLIArgumentParsing:
    """Test CLI argument parsing for benchmark script."""

    def test_parse_benchmark_args_with_task_ids(self):
        """Test parsing CLI args with task IDs."""
        from scripts.benchmark_evolution import parse_benchmark_args

        args = parse_benchmark_args(
            [
                "--tasks",
                "00576224,007bbfb7",
                "--output-dir",
                "results/",
                "--experiment-name",
                "test",
            ]
        )

        assert args.tasks == "00576224,007bbfb7"
        assert args.output_dir == "results/"
        assert args.experiment_name == "test"

    def test_parse_benchmark_args_with_defaults(self):
        """Test that default arguments are set correctly."""
        from scripts.benchmark_evolution import parse_benchmark_args

        # Must provide required task selection argument
        args = parse_benchmark_args(
            [
                "--tasks",
                "00576224",
                "--output-dir",
                "results/",
                "--experiment-name",
                "test",
            ]
        )

        assert args.max_generations == 5
        assert args.sandbox_mode == "multiprocess"
        assert args.model == "gemini-2.5-flash-lite"
        assert args.programmer_temperature == 0.3
        assert args.refiner_temperature == 0.4
        assert args.timeout_eval == 5
        assert args.timeout_llm == 60
        assert args.use_cache is True

    def test_parse_benchmark_args_with_random_sample(self):
        """Test parsing with random sample flag."""
        from scripts.benchmark_evolution import parse_benchmark_args

        args = parse_benchmark_args(
            [
                "--random-sample",
                "10",
                "--training-data",
                "data/training.json",
                "--output-dir",
                "results/",
                "--experiment-name",
                "test",
            ]
        )

        assert args.random_sample == 10
        assert args.training_data == "data/training.json"


class TestResumeCapability:
    """Test resume capability for interrupted runs."""

    def test_get_completed_task_ids(self, tmp_path):
        """Test identifying already-completed tasks."""
        from scripts.benchmark_evolution import get_completed_task_ids

        output_dir = tmp_path / "results"
        output_dir.mkdir()

        # Create some result files
        (output_dir / "task_00576224.json").write_text(json.dumps({"task_id": "00576224"}))
        (output_dir / "task_007bbfb7.json").write_text(json.dumps({"task_id": "007bbfb7"}))

        completed = get_completed_task_ids(str(output_dir))

        assert len(completed) == 2
        assert "00576224" in completed
        assert "007bbfb7" in completed

    def test_filter_remaining_tasks(self):
        """Test filtering out already-completed tasks."""
        from scripts.benchmark_evolution import filter_remaining_tasks

        all_tasks = ["00576224", "007bbfb7", "025d127b", "009d5c81"]
        completed = {"007bbfb7", "025d127b"}

        remaining = filter_remaining_tasks(all_tasks, completed)

        assert len(remaining) == 2
        assert "00576224" in remaining
        assert "009d5c81" in remaining
        assert "007bbfb7" not in remaining

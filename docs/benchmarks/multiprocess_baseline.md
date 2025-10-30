# Benchmark Report: multiprocess_baseline

**Date**: 2025-10-30T08:39:34.195274+00:00
**Git Commit**: `3e20c1ad0ed6a1234b8827a5f66c6f2eba8d13a2`
**Tasks**: 15

## Configuration

- **max_generations**: 5
- **model**: gemini-2.5-flash-lite
- **programmer_temperature**: 0.3
- **refiner_temperature**: 0.4
- **sandbox_mode**: multiprocess
- **seed**: None
- **target_fitness**: None
- **timeout_eval**: 5
- **timeout_llm**: 60
- **use_cache**: False

## Overall Results

- **Total Tasks**: 15
- **Successful**: 15
- **Failed**: 0
- **Success Rate**: 100.0%

### Fitness Metrics

- **Average**: 0.33
- **Median**: 0.00

### Evolution Metrics

- **Avg Generations**: 5.00
- **Avg Time per Task**: 25.5s
- **Total Time**: 382.3s

### Error Distribution

- **logic**: 162
- **syntax**: 15
- **validation**: 20

## Task-by-Task Results

| Task ID | Success | Final Fitness | Generations | Time (s) |
|---------|---------|---------------|-------------|----------|
| 00576224 | ✅ | 2.0 | 5 | 26.4 |
| 007bbfb7 | ✅ | 0.0 | 5 | 26.0 |
| 025d127b | ✅ | 0.0 | 5 | 14.0 |
| 045e512c | ✅ | 0.0 | 5 | 25.9 |
| 0520fde7 | ✅ | 0.0 | 5 | 40.3 |
| 05269061 | ✅ | 0.0 | 5 | 15.0 |
| 05a7bcf2 | ✅ | 0.0 | 5 | 29.6 |
| 0607ce86 | ✅ | 0.0 | 5 | 14.2 |
| 06df4c85 | ✅ | 0.0 | 5 | 20.2 |
| 070dd51e | ✅ | 0.0 | 5 | 17.4 |
| 08ed6ac7 | ✅ | 2.0 | 5 | 12.2 |
| 09629e4f | ✅ | 0.0 | 5 | 26.4 |
| 09c534e7 | ✅ | 0.0 | 5 | 44.4 |
| 0a1d4ef5 | ✅ | 1.0 | 5 | 44.5 |
| 0bb8deee | ✅ | 0.0 | 5 | 25.9 |

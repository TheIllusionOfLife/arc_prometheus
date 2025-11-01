# Phase 4: Validation-First Optimization Strategy

**Date:** November 02, 2025
**Status:** Ready for Execution
**Estimated Duration:** 1 week (vs 3-4 days for blind distillation)
**Expected Outcome:** Validated approach with proven improvements

---

## Executive Summary

### **Why This Approach is Better**

**Original Plan (Phase 4b):**
- ❌ Assumes Gemini performs well (not validated)
- ❌ Assumes distillation transfers performance (risky)
- ❌ Costs $50-75 + 3-4 days without knowing if it works
- ❌ **Fatal flaw:** If Gemini scores 5%, distillation is pointless

**Validation-First Approach:**
- ✅ Validates Gemini baseline BEFORE investing in distillation
- ✅ Quick iteration (1-hour tests vs 8-12 hour runs)
- ✅ Tests improvements incrementally (Active Inference, hyperparameters)
- ✅ Optimizes source before copying (improve Gemini → then distill)
- ✅ Cost-effective ($5-10 validation vs $50-75 blind collection)

**Scientific Method:**
```
Hypothesis → Quick Test → Measure → Optimize → Validate → Scale
```

vs. blind approach:
```
Assume → Build → Hope it works → Expensive failure
```

---

## The Validation-First Plan

### **Phase 1: Gemini Baseline Validation** (1-2 hours)

**Objective:** Determine if Gemini performs well enough to justify knowledge distillation.

#### **Step 1.1: Quick Test (10 tasks)** (15 minutes)

**Purpose:** Get rough time/cost estimates before committing to larger test.

```bash
# Run on 10 random evaluation tasks
python scripts/benchmark_evolution.py \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --random-sample 10 \
  --output-dir results/gemini_baseline_quick/ \
  --experiment-name "gemini_baseline_10tasks" \
  --max-generations 3 \
  --population-size 2 \
  --use-analyst
```

**Measurements:**
- ⏱️ **Time per task:** (total_time / 10) → estimate for 120 tasks
- 📊 **Score:** train_accuracy, test_accuracy, avg_fitness
- 💰 **Cost:** (API_calls × $0.01) → estimate for full run
- 🔍 **Success rate:** tasks_with_fitness_>_0 / 10

**Example output:**
```
10 tasks completed in 6 minutes
→ 120 tasks would take: 72 minutes (too long for 1h test)
→ Adjust to 40 tasks for 1-hour validation

Average fitness: 4.2
Test accuracy: 15% (3 correct / 20 test examples)
API cost: $1.50 (150 calls)
→ 120 tasks would cost: ~$18
```

#### **Step 1.2: Calculate Optimal Sample Size** (5 minutes)

**Goal:** Find sample size that finishes in ~1 hour and gives statistically valid score.

**Formula:**
```python
time_per_task = total_time_10_tasks / 10
target_duration = 3600  # 1 hour in seconds
optimal_sample = int(target_duration / time_per_task)

# Cap at 50 tasks (statistically sufficient)
sample_size = min(optimal_sample, 50)
```

**Statistical validity:**
- 30+ tasks: 95% confidence (standard for experiments)
- 40-50 tasks: Excellent confidence
- 120 tasks: Overkill for validation (save for final run)

#### **Step 1.3: Run Baseline Validation** (1 hour)

```bash
# Run adjusted sample size (e.g., 40 tasks)
python scripts/benchmark_evolution.py \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --random-sample 40 \
  --seed 42 \
  --output-dir results/gemini_baseline_validation/ \
  --experiment-name "gemini_baseline_40tasks" \
  --max-generations 3 \
  --population-size 2 \
  --use-analyst
```

**Success Criteria:**
```python
# Calculate score from results
test_correct = sum(task["test_correct"] for task in results)
test_total = sum(len(task["test_examples"]) for task in results)
score = test_correct / test_total * 100

print(f"Gemini Baseline Score: {score:.1f}%")
```

#### **Decision Gate 1: Is Gemini Good Enough?**

**✅ Score ≥8%:** Proceed to Phase 2 (Active Inference testing)
- Reasoning: 8% baseline + 5% Active Inference = 13% (competitive)
- Action: Test Active Inference on same 40 tasks

**⚠️ Score 5-8%:** Optimize before Active Inference
- Reasoning: Need stronger baseline before testing incremental improvements
- Action: Phase 2b (Hyperparameter Tuning) first

**❌ Score <5%:** Fundamental rethink needed
- Reasoning: Even with Active Inference (+5%), still below competitive threshold
- Action: Analyze failure modes, consider alternative approaches
- Options:
  - Improve Analyst prompt engineering
  - Try different LLM (e.g., gemini-2.0-flash-thinking-exp)
  - Increase generations significantly (5 → 10)
  - Manual error analysis of failures

---

### **Phase 2a: Active Inference Testing** (2-3 hours)

**Prerequisite:** Gemini baseline ≥8%

**Objective:** Test if per-task fine-tuning improves performance (Jack Cole's 34% SOTA approach).

#### **What is Active Inference?**

**Key Idea:**
- For each test task, augment its 3 training examples → 30+ variations
- Generate diverse examples while preserving transformation rule
- Give LLM more "experience" with the specific task pattern

**Augmentation Techniques:**
```python
# scripts/augment_task_examples.py
def augment_examples(task_data, num_variations=10):
    """
    Generate variations of training examples.

    Techniques:
    1. Rotations: 90°, 180°, 270°
    2. Flips: horizontal, vertical
    3. Color swaps: permute colors while preserving pattern
    4. Translations: shift grid (if applicable)
    5. Scale: resize grid (if pattern allows)

    CRITICAL: Preserve transformation rule invariance!
    """
    augmented = []

    for example in task_data["train"]:
        # Original
        augmented.append(example)

        # Rotations (if transformation is rotation-invariant)
        for k in [1, 2, 3]:  # 90°, 180°, 270°
            augmented.append({
                "input": np.rot90(example["input"], k).tolist(),
                "output": np.rot90(example["output"], k).tolist()
            })

        # Flips
        augmented.append({
            "input": np.fliplr(example["input"]).tolist(),
            "output": np.fliplr(example["output"]).tolist()
        })
        augmented.append({
            "input": np.flipud(example["input"]).tolist(),
            "output": np.flipud(example["output"]).tolist()
        })

        # Color permutations (up to 10 colors: 0-9)
        for perm in generate_color_permutations(limit=3):
            augmented.append({
                "input": apply_color_map(example["input"], perm),
                "output": apply_color_map(example["output"], perm)
            })

    # Return exactly num_variations * original_count examples
    target_count = num_variations * len(task_data["train"])
    return augmented[:target_count]
```

#### **Step 2a.1: Implement Augmentation Logic** (1 hour)

**Files to create:**
1. `src/arc_prometheus/cognitive_cells/augmentation.py` - Core augmentation functions
2. Update `evolution_loop.py` - Add `use_active_inference` parameter
3. Update `benchmark_evolution.py` - Add `--use-active-inference` CLI flag

**Integration:**
```python
# In evolution_loop.py
def run_evolution_loop(
    task_json_path: str,
    use_active_inference: bool = False,
    augmentation_factor: int = 10,
    ...
):
    # Load task
    task = load_task(task_json_path)

    if use_active_inference:
        # Augment training examples
        task["train"] = augment_examples(task, num_variations=augmentation_factor)
        print(f"Active Inference: Augmented {len(task['train'])} examples")

    # Rest of evolution loop unchanged
    ...
```

#### **Step 2a.2: Run Active Inference Test** (1 hour)

**CRITICAL:** Use same 40 tasks with same seed for fair comparison!

```bash
# Run SAME 40 tasks with Active Inference enabled
python scripts/benchmark_evolution.py \
  --training-data data/arc-prize-2025/arc-agi_evaluation_challenges_merged.json \
  --random-sample 40 \
  --seed 42 \
  --output-dir results/gemini_active_inference/ \
  --experiment-name "gemini_active_40tasks" \
  --max-generations 3 \
  --population-size 2 \
  --use-analyst \
  --use-active-inference \
  --augmentation-factor 10
```

#### **Step 2a.3: Compare Results** (15 minutes)

**Create comparison script:**
```python
# scripts/compare_results.py
import json

def compare_experiments(baseline_dir, active_dir):
    baseline = load_aggregate_stats(baseline_dir)
    active = load_aggregate_stats(active_dir)

    print("=" * 60)
    print("GEMINI PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'Active':>12} {'Δ':>12}")
    print("-" * 60)

    metrics = [
        ("Test Accuracy", "test_accuracy_pct"),
        ("Train Accuracy", "train_accuracy_pct"),
        ("Avg Fitness", "avg_fitness"),
        ("Tasks Solved (fitness>0)", "tasks_with_positive_fitness"),
        ("Perfect Solvers (fitness=13)", "perfect_solvers"),
        ("Avg Runtime (sec/task)", "avg_runtime_per_task"),
        ("Total API Cost", "total_api_cost"),
    ]

    for name, key in metrics:
        baseline_val = baseline[key]
        active_val = active[key]
        delta = active_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val > 0 else 0

        print(f"{name:<30} {baseline_val:>12.2f} {active_val:>12.2f} {delta:>+12.2f} ({delta_pct:+.1f}%)")

    print("=" * 60)

    # Decision recommendation
    improvement = active["test_accuracy_pct"] - baseline["test_accuracy_pct"]

    if improvement >= 2.0:
        print("✅ RECOMMENDATION: Keep Active Inference (+{:.1f}% improvement)".format(improvement))
    elif improvement > 0:
        print("⚠️  MARGINAL: Consider cost/benefit (+{:.1f}% for {}x runtime)".format(
            improvement, active["avg_runtime_per_task"] / baseline["avg_runtime_per_task"]))
    else:
        print("❌ NO BENEFIT: Skip Active Inference (-{:.1f}% change)".format(improvement))

if __name__ == "__main__":
    compare_experiments(
        "results/gemini_baseline_validation/",
        "results/gemini_active_inference/"
    )
```

**Run comparison:**
```bash
python scripts/compare_results.py
```

#### **Decision Gate 2: Does Active Inference Help?**

**✅ Improvement ≥2%:** Keep Active Inference, proceed to Phase 3
- Example: 8% → 10% (competitive threshold reached!)
- Action: Use Active Inference in final Kaggle submission

**⚠️ Improvement 0-2%:** Marginal benefit
- Consider: Does 1% improvement justify 2x runtime?
- If yes: Keep for final submission
- If no: Skip, use baseline Gemini

**❌ No improvement or worse:** Skip Active Inference
- Possible causes:
  - Augmentation breaks task-specific patterns
  - LLM overfits to augmented data
  - Increased noise from too many examples
- Action: Use baseline Gemini, test other optimizations

---

### **Phase 2b: Hyperparameter Tuning** (if baseline 5-8% or Active Inference doesn't help)

**Objective:** Find optimal configuration before committing to full distillation.

#### **Parameters to Test:**

**1. Generations (currently 3):**
```bash
# Test 5 generations
python scripts/benchmark_evolution.py \
  --random-sample 20 \
  --seed 42 \
  --max-generations 5
```

**2. Population Size (currently 2):**
```bash
# Test population=3
python scripts/benchmark_evolution.py \
  --random-sample 20 \
  --seed 42 \
  --population-size 3
```

**3. Temperature (currently 0.7):**
```bash
# Test higher temperature for more creativity
python scripts/benchmark_evolution.py \
  --random-sample 20 \
  --seed 42 \
  --programmer-temperature 0.8
```

**4. Model (currently gemini-1.5-flash):**
```bash
# Test thinking model
python scripts/benchmark_evolution.py \
  --random-sample 20 \
  --seed 42 \
  --model gemini-2.0-flash-thinking-exp
```

#### **Grid Search Strategy:**

**Quick tests on 20 tasks each (15-20 min per config):**
```python
configs = [
    {"generations": 5, "population": 2},
    {"generations": 3, "population": 3},
    {"temperature": 0.8, "generations": 3},
    {"model": "gemini-2.0-flash-thinking-exp"},
]

best_config = None
best_score = 0

for config in configs:
    score = run_test(config, sample_size=20)
    if score > best_score:
        best_score = score
        best_config = config
```

**Then validate best config on 40 tasks:**
```bash
python scripts/benchmark_evolution.py \
  --random-sample 40 \
  --seed 42 \
  --max-generations {best_generations} \
  --population-size {best_population} \
  ...
```

---

### **Phase 3: Knowledge Distillation** (ONLY if Gemini ≥8%)

**Prerequisites:**
- ✅ Gemini baseline validated ≥8%
- ✅ Best configuration identified (baseline or with Active Inference)
- ✅ Competition deadline allows 3-4 days

**If prerequisites met, proceed with original Phase 4b plan:**

See `plan_phase4b_distillation.md` for full details. Summary:

1. **Collect training data** (400 tasks, 8-12h, $50-75)
   - Use best config from Phase 2
   - Collect successful Gemini outputs

2. **Fine-tune Code Gemma** (2-4h)
   - Train on Gemini's successful solvers
   - Use LoRA for efficiency

3. **Deploy to Kaggle** (7h)
   - Upload fine-tuned model
   - Run on 240 test tasks
   - Submit to competition

**Expected improvement:**
- Gemini: 8-13% (validated)
- Code Gemma (distilled): 5-10% (70-80% of Gemini's performance)
- Still competitive if Gemini ≥10%

---

## Timeline Summary

### **Fast Path (if Gemini is good and Active Inference helps):**
```
Day 1 (3-4 hours):
  Phase 1: Gemini baseline validation (10 + 40 tasks)
  Phase 2a: Active Inference test (40 tasks)
  Decision: Gemini 8% → 10% with Active Inference ✅

Day 2-5 (3-4 days):
  Phase 3: Knowledge distillation (if time permits)
  OR: Use Gemini directly with Active Inference for submission

Result: Validated 10% solution in 4 hours, skip risky distillation
```

### **Optimization Path (if baseline needs improvement):**
```
Day 1 (3-4 hours):
  Phase 1: Gemini baseline validation (6%)
  Phase 2b: Hyperparameter tuning (find best config → 8%)

Day 2 (2 hours):
  Phase 2a: Active Inference test (8% → 10%)

Day 3-6:
  Phase 3: Knowledge distillation (if time permits)

Result: Optimized to 10% before committing to distillation
```

### **Deadline-Driven Decision:**
```
Competition deadline: November 3, 2025 (1 day!)

Option A: Skip distillation, use Gemini directly
  - Pro: Immediate validated solution
  - Con: No offline Kaggle deployment

Option B: Rush distillation (if Gemini ≥10%)
  - Pro: Offline deployment, potential for improvement
  - Con: Risky with 1-day deadline

Recommendation: Validate first (4 hours), decide based on results
```

---

## Success Criteria

### **Phase 1 Success:**
- [ ] 10-task quick test completed
- [ ] Time/cost estimates calculated
- [ ] 40-task validation run completed
- [ ] Gemini baseline score measured
- [ ] Decision gate criteria met (≥8% or pivot to optimization)

### **Phase 2a Success:**
- [ ] Augmentation logic implemented and tested
- [ ] Active Inference run on same 40 tasks
- [ ] Comparison shows ≥2% improvement
- [ ] Decision on keeping Active Inference made

### **Phase 2b Success (if needed):**
- [ ] At least 3 hyperparameter configs tested
- [ ] Best config identified
- [ ] Improvement over baseline demonstrated

### **Phase 3 Success (if attempted):**
- [ ] See `plan_phase4b_distillation.md` success criteria

---

## Resource Estimates

### **Phase 1 (Validation):**
- **Time:** 1-2 hours (10 tasks + 40 tasks)
- **Cost:** $5-10 (Gemini API)
- **Output:** Validated baseline score

### **Phase 2a (Active Inference):**
- **Time:** 2-3 hours (implementation + test)
- **Cost:** $5-10 (same 40 tasks)
- **Output:** Improvement measurement

### **Phase 2b (Hyperparameter Tuning):**
- **Time:** 2-4 hours (4-6 configs × 20 tasks each)
- **Cost:** $10-15 (Gemini API)
- **Output:** Best configuration

### **Phase 3 (Knowledge Distillation):**
- **Time:** 3-4 days
- **Cost:** $50-75 (Gemini API) + free GPU (Kaggle)
- **Output:** Fine-tuned Code Gemma

### **Total (all phases):**
- **Time:** 1 week max
- **Cost:** $70-100
- **vs. blind distillation:** Same cost, but validated at each step

---

## Risk Mitigation

### **Risk 1: Gemini performs poorly (<5%)**
- **Impact:** Knowledge distillation would be pointless
- **Mitigation:** Discover this in Phase 1 (4 hours) vs after Phase 4b (3-4 days)
- **Fallback:** Optimize Gemini first, or pivot to different approach

### **Risk 2: Active Inference doesn't help**
- **Impact:** Extra runtime without benefit
- **Mitigation:** Test on 40 tasks (1 hour) before full run
- **Fallback:** Use baseline Gemini

### **Risk 3: Competition deadline too tight**
- **Impact:** Can't complete knowledge distillation
- **Mitigation:** Validated Gemini solution ready after Phase 1-2
- **Fallback:** Submit Gemini results directly (if allowed) or document findings

### **Risk 4: Hyperparameters don't improve baseline**
- **Impact:** Stuck at 5-8% range
- **Mitigation:** Quick tests (20 tasks) before committing
- **Fallback:** Fundamental approach may need rethinking

---

## Next Steps

### **Immediate (Today):**

1. **Create validation scripts** (30 minutes)
   - `scripts/validate_gemini_baseline.py`
   - `scripts/compare_results.py`
   - Update `benchmark_evolution.py` with `--use-active-inference`

2. **Run Phase 1 quick test** (15 minutes)
   - 10 tasks to get time/cost estimates
   - Calculate optimal sample size

3. **Run Phase 1 full validation** (1 hour)
   - 40 tasks for statistically valid score
   - Make decision: optimize or proceed

### **Based on Results:**

**If Gemini ≥8%:**
- Proceed to Phase 2a (Active Inference)
- 2-3 hours to test and compare
- Decision on final approach

**If Gemini 5-8%:**
- Proceed to Phase 2b (Hyperparameter Tuning)
- 2-4 hours to find best config
- Then test Active Inference

**If Gemini <5%:**
- Analyze failure modes
- Consider fundamental changes
- May need different approach entirely

---

## Conclusion

**This validation-first approach is superior because:**

1. ✅ **De-risks expensive operations** (validate before $50-75 spend)
2. ✅ **Enables quick iteration** (1-hour tests vs 8-12 hour runs)
3. ✅ **Provides decision gates** (pivot based on data, not assumptions)
4. ✅ **Optimizes source before copying** (improve Gemini → then distill)
5. ✅ **Deadline-friendly** (can stop after Phase 1-2 with validated solution)

**vs. original plan:**
- ❌ Assumes Gemini works (risky)
- ❌ No checkpoints to pivot (all-or-nothing)
- ❌ Expensive failure mode ($50-75 wasted if Gemini scores 5%)

**The scientific approach wins: measure → optimize → scale.**

---

**Status:** Ready for execution
**Next:** Create validation scripts and run Phase 1 quick test
**Owner:** TBD
**Priority:** CRITICAL (competition deadline November 3)

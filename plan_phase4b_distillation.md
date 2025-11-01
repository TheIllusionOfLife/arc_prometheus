# Phase 4b: Knowledge Distillation - Detailed Implementation Plan

**Date:** November 02, 2025
**Status:** Ready for Execution (Waiting for Task 4.4 completion)
**Prerequisites:** Baseline Kaggle submission complete (Task 4.4)
**Estimated Duration:** 3-4 days
**Expected Improvement:** +5-10% over baseline

---

## Executive Summary

**Goal:** Fine-tune Code Gemma 7B on successful Gemini-generated solvers to improve baseline Kaggle score.

**Why Critical:** Baseline Code Gemma 7B is too weak (fitness=0 on test tasks). Knowledge distillation transfers Gemini's problem-solving ability to the local model, enabling competitive performance.

**Expected Impact:** +5-10% improvement could move from 15th place → 5th place on leaderboard (competitive threshold: 10%).

---

## Task Breakdown

### **Task 4b.1: Collect Gemini Training Data** (1 day)

**Objective:** Generate high-quality training data from Gemini API on all 400 ARC training tasks.

#### Step 1.1: Create Data Collection Script

**File:** `scripts/collect_distillation_data.py`

```python
"""
Collect Gemini-generated solvers for knowledge distillation.
Runs full AI Civilization pipeline (Analyst → Programmer → Refiner) on training set.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from arc_prometheus.cognitive_cells import Analyst, Programmer, Refiner
from arc_prometheus.evolutionary_engine import run_evolution_loop
from arc_prometheus.crucible import load_task

def collect_training_data(
    training_data_path: str,
    output_path: str,
    max_generations: int = 5,
    min_fitness: float = 1.0,
    use_analyst: bool = True,
    verbose: bool = True
):
    """
    Collect successful Gemini solvers for distillation.

    Args:
        training_data_path: Path to arc-agi_training_challenges.json
        output_path: Output JSON file for distillation dataset
        max_generations: Max evolution generations per task
        min_fitness: Minimum fitness to include solver in dataset
        use_analyst: Use AI Civilization mode (recommended)
        verbose: Show progress

    Output Format:
        [
            {
                "task_id": "00576224",
                "analyst_spec": "...",
                "solver_code": "def solve(...)...",
                "fitness": 13.0,
                "train_correct": 3,
                "test_correct": 1
            },
            ...
        ]
    """
    with open(training_data_path) as f:
        tasks = json.load(f)

    distillation_dataset = []
    successful_tasks = 0

    for task_id, task_data in tqdm(tasks.items(), desc="Collecting training data"):
        try:
            # Run evolution loop
            result = run_evolution_loop(
                task_json_path=f"temp_{task_id}.json",  # Temp file
                max_generations=max_generations,
                target_fitness=13.0,  # Perfect score
                use_analyst=use_analyst,
                verbose=False
            )

            # Get best solver from final generation
            best_solver = result["generations"][-1]

            # Only include successful solvers (fitness > min_fitness)
            if best_solver["fitness"] >= min_fitness:
                distillation_dataset.append({
                    "task_id": task_id,
                    "analyst_spec": best_solver.get("analyst_spec", ""),
                    "solver_code": best_solver["code"],
                    "fitness": best_solver["fitness"],
                    "train_correct": best_solver["train_correct"],
                    "test_correct": best_solver["test_correct"]
                })
                successful_tasks += 1

                if verbose:
                    print(f"✅ {task_id}: fitness={best_solver['fitness']:.1f}")
            else:
                if verbose:
                    print(f"❌ {task_id}: fitness={best_solver['fitness']:.1f} (below threshold)")

        except Exception as e:
            if verbose:
                print(f"❌ {task_id}: ERROR - {e}")
            continue

    # Save dataset
    with open(output_path, "w") as f:
        json.dump(distillation_dataset, f, indent=2)

    print(f"\n✅ Collection complete!")
    print(f"Successful tasks: {successful_tasks}/{len(tasks)} ({successful_tasks/len(tasks)*100:.1f}%)")
    print(f"Dataset saved to: {output_path}")

    return distillation_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Gemini training data for distillation")
    parser.add_argument("--training-data", required=True, help="Path to training challenges JSON")
    parser.add_argument("--output", default="data/distillation_dataset.json", help="Output dataset path")
    parser.add_argument("--max-generations", type=int, default=5, help="Max evolution generations")
    parser.add_argument("--min-fitness", type=float, default=1.0, help="Minimum fitness threshold")
    parser.add_argument("--no-analyst", action="store_true", help="Disable Analyst (Direct mode)")

    args = parser.parse_args()

    collect_training_data(
        training_data_path=args.training_data,
        output_path=args.output,
        max_generations=args.max_generations,
        min_fitness=args.min_fitness,
        use_analyst=not args.no_analyst
    )
```

#### Step 1.2: Run Data Collection

```bash
# Full collection (400 tasks, ~8-12 hours)
python scripts/collect_distillation_data.py \
  --training-data data/arc-prize-2025/arc-agi_training_challenges.json \
  --output data/distillation_dataset.json \
  --max-generations 5 \
  --min-fitness 1.0

# Expected: 80-120 successful solvers (20-30% success rate)
```

#### Step 1.3: Convert to Fine-tuning Format

**File:** `scripts/prepare_finetuning_data.py`

```python
"""
Convert distillation dataset to Hugging Face fine-tuning format.
"""
import json
from datasets import Dataset

def convert_to_instruction_format(distillation_dataset: list) -> list:
    """
    Convert to instruction-response pairs for fine-tuning.

    Format:
        Instruction: "Analyze and solve this ARC puzzle: [task examples]"
        Response: "[Analyst spec]\n\n[Solver code]"
    """
    formatted_data = []

    for entry in distillation_dataset:
        instruction = f"""Analyze and solve this ARC puzzle.

Training examples:
[Format task examples here]

Provide:
1. Analysis of the transformation pattern
2. Python solver function
"""

        response = f"""{entry['analyst_spec']}

```python
{entry['solver_code']}
```
"""

        formatted_data.append({
            "instruction": instruction,
            "response": response
        })

    return formatted_data

if __name__ == "__main__":
    with open("data/distillation_dataset.json") as f:
        dataset = json.load(f)

    formatted = convert_to_instruction_format(dataset)

    # Save as Hugging Face dataset
    hf_dataset = Dataset.from_list(formatted)
    hf_dataset.save_to_disk("data/distillation_hf_dataset")

    print(f"✅ Fine-tuning dataset prepared: {len(formatted)} examples")
```

**Success Criteria for Task 4b.1:**
- [ ] Data collection script created and tested
- [ ] 80-120 successful solvers collected (20-30% of 400 tasks)
- [ ] Distillation dataset saved to `data/distillation_dataset.json`
- [ ] Fine-tuning format conversion working
- [ ] Dataset split: 80% train (64-96 examples) / 20% validation (16-24 examples)

---

### **Task 4b.2: Fine-tune Code Gemma** (1-2 days)

**Objective:** Fine-tune Code Gemma 7B on collected Gemini outputs using LoRA/QLoRA for efficiency.

#### Step 2.1: Set Up Fine-tuning Environment

**Requirements:**
- GPU with ≥24GB VRAM (recommended: A100, L4x4 on Kaggle)
- Libraries: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`

```bash
# Install fine-tuning dependencies
pip install transformers peft bitsandbytes accelerate datasets

# Or add to pyproject.toml:
# [project.optional-dependencies]
# finetune = ["transformers", "peft", "bitsandbytes", "accelerate", "datasets"]
```

#### Step 2.2: Create Fine-tuning Script

**File:** `scripts/finetune_codegemma.py`

```python
"""
Fine-tune Code Gemma 7B using LoRA on distillation dataset.
Uses QLoRA (4-bit quantization) for memory efficiency.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

def setup_lora_model(model_path: str, lora_r: int = 16, lora_alpha: int = 32):
    """
    Load Code Gemma and apply LoRA configuration.

    LoRA Config:
        - r=16: Low-rank dimension (controls trainable params)
        - alpha=32: Scaling factor
        - Target modules: q_proj, v_proj (attention layers)
        - Dropout: 0.05 for regularization
    """
    # Load base model in 4-bit (QLoRA)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should be ~1-2% of total params

    return model

def finetune_codegemma(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """
    Fine-tune Code Gemma on distillation dataset.

    Args:
        model_path: Path to base Code Gemma model
        dataset_path: Path to Hugging Face dataset
        output_dir: Output directory for fine-tuned model
        num_epochs: Training epochs (3 recommended)
        batch_size: Batch size (4 for 24GB GPU)
        learning_rate: Learning rate (2e-4 typical for LoRA)
    """
    # Load model and tokenizer
    model = setup_lora_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Tokenize function
    def tokenize_function(examples):
        # Combine instruction + response
        texts = [
            f"{inst}\n\n{resp}"
            for inst, resp in zip(examples["instruction"], examples["response"])
        ]
        return tokenizer(texts, truncation=True, max_length=2048, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_steps=100,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Fine-tune!
    print("🚀 Starting fine-tuning...")
    trainer.train()

    # Save final model
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"✅ Fine-tuning complete! Model saved to: {output_dir}/final")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Code Gemma on distillation data")
    parser.add_argument("--model-path", default="models/codegemma-7b/model", help="Base model path")
    parser.add_argument("--dataset-path", default="data/distillation_hf_dataset", help="Dataset path")
    parser.add_argument("--output-dir", default="models/codegemma-7b-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    finetune_codegemma(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
```

#### Step 2.3: Run Fine-tuning

```bash
# Local fine-tuning (if you have GPU)
python scripts/finetune_codegemma.py \
  --model-path models/codegemma-7b/model \
  --dataset-path data/distillation_hf_dataset \
  --output-dir models/codegemma-7b-finetuned \
  --epochs 3 \
  --batch-size 4

# OR: Use Kaggle Notebook for free GPU access
# - Upload distillation dataset to Kaggle dataset
# - Create notebook with fine-tuning script
# - Run on L4x4 (96GB VRAM, plenty for QLoRA)
```

**Expected Training Time:**
- 80-120 examples × 3 epochs ≈ 2-4 hours on A100/L4x4
- QLoRA reduces memory: 15.93 GB → ~8-10 GB (fits in 24GB GPU)

**Success Criteria for Task 4b.2:**
- [ ] Fine-tuning script created and tested
- [ ] Model fine-tuned successfully (3 epochs)
- [ ] Validation loss decreasing across epochs
- [ ] Fine-tuned model saved to `models/codegemma-7b-finetuned/`
- [ ] LoRA adapters merged back into base model

---

### **Task 4b.3: Update Kaggle Notebook** (0.5 days)

**Objective:** Replace base Code Gemma with fine-tuned version in Kaggle submission notebook.

#### Step 3.1: Upload Fine-tuned Model to Kaggle

```bash
# 1. Download fine-tuned model from local/Kaggle
# 2. Upload to Kaggle Datasets:
#    - Title: "Code Gemma 7B Fine-tuned on ARC (Gemini Distillation)"
#    - Files: models/codegemma-7b-finetuned/final/*
#    - Size: ~16 GB (same as base model)
```

#### Step 3.2: Update Notebook Cell 2

**Change:**
```python
# OLD (Cell 2 - Load Base Model)
MODEL_PATH = "/kaggle/input/codegemma-7b/model"

# NEW (Cell 2 - Load Fine-tuned Model)
MODEL_PATH = "/kaggle/input/codegemma-7b-finetuned/final"
```

**That's it!** The rest of the notebook remains unchanged.

#### Step 3.3: Test on Evaluation Set First

```bash
# Before full submission, test on evaluation set (120 tasks)
# Set TEST_MODE = True, run on eval set
# Expected: Fitness improvement vs baseline
```

**Success Criteria for Task 4b.3:**
- [ ] Fine-tuned model uploaded to Kaggle dataset
- [ ] Notebook Cell 2 updated with new MODEL_PATH
- [ ] Test run on 5-10 tasks successful (no errors)
- [ ] Fitness improvement visible vs baseline

---

### **Task 4b.4: Full Kaggle Submission** (0.5 days)

**Objective:** Submit improved model to competition and measure improvement.

#### Step 4.1: Run Full Submission

```python
# In Kaggle notebook Cell 5:
TEST_MODE = False  # Run all 240 tasks
population_size = 2  # Validated safe for 12h limit
max_generations = 3  # Balanced speed/quality
```

**Expected Runtime:** ~7 hours (validated in PR #47)

#### Step 4.2: Submit and Compare

1. Download `submission.json` from notebook
2. Submit to ARC Prize 2025 competition
3. Wait for leaderboard score
4. Compare: baseline vs fine-tuned

**Expected Results:**
- Baseline (Code Gemma 7B): 3-8%
- Fine-tuned (Gemini distillation): 8-18% (+5-10%)
- Competitive threshold: 10% (top 10 placement)

**Success Criteria for Task 4b.4:**
- [ ] Full submission complete (240 tasks)
- [ ] Submission accepted by Kaggle
- [ ] Leaderboard score received
- [ ] Improvement ≥5% over baseline
- [ ] Documentation updated with results

---

## Risk Mitigation

### Risk 1: Low Collection Success Rate (<20%)
**Mitigation:**
- Lower min_fitness threshold (1.0 → 0.5)
- Increase max_generations (5 → 10)
- Use evaluation set (120 tasks) as backup training data

### Risk 2: Fine-tuning Overfits
**Mitigation:**
- Reduce epochs (3 → 2)
- Increase LoRA dropout (0.05 → 0.1)
- Monitor validation loss carefully

### Risk 3: Fine-tuned Model Performs Worse
**Mitigation:**
- Keep base model submission as fallback
- A/B test: submit both versions
- Analyze: which tasks improved vs degraded?

### Risk 4: Kaggle Runtime Exceeds 12h
**Mitigation:**
- Reduce population_size (2 → 1)
- Reduce max_generations (3 → 2)
- Pre-validated: pop=2, gen=3 takes ~7h (5h buffer)

---

## Timeline

**Day 1:** Task 4b.1 - Collect training data (8-12 hours runtime)
**Day 2:** Task 4b.2 - Fine-tune model (2-4 hours runtime)
**Day 3:** Task 4b.3 - Update notebook and test (2-3 hours)
**Day 4:** Task 4b.4 - Full submission and comparison (7 hours runtime + analysis)

**Total:** 3-4 days (mostly waiting for long-running processes)

---

## Success Metrics

**Must Have:**
- [ ] ≥80 successful solvers collected
- [ ] Fine-tuned model trained successfully
- [ ] Kaggle submission accepted
- [ ] Leaderboard score ≥ baseline

**Should Have:**
- [ ] ≥5% improvement over baseline
- [ ] Score ≥10% (competitive threshold)
- [ ] <12h runtime on full submission

**Nice to Have:**
- [ ] ≥10% improvement over baseline
- [ ] Score ≥15% (top 5 placement)
- [ ] Detailed analysis of which task types improved

---

## Next Steps After Phase 4b

If Phase 4b achieves 10%+ score:
- **Phase 4c:** Active Inference (+5-10% more)
- **Phase 4d:** Multi-Agent Active Inference (+2-5% more)
- **Target:** 20-30% competitive score

If Phase 4b score <10%:
- Analyze failure modes
- Collect more diverse training data
- Try larger model (Code Gemma 13B)
- Consider alternative distillation approaches

---

## Resources

**Compute Requirements:**
- Data collection: 8-12 hours on Gemini API (~$20-30 cost)
- Fine-tuning: 2-4 hours on A100/L4x4 GPU (free on Kaggle)
- Full submission: 7 hours on L4x4 GPU (free on Kaggle)

**Storage Requirements:**
- Distillation dataset: ~5-10 MB (JSON)
- Fine-tuned model: ~16 GB (same as base model)
- Total: ~32 GB (base + fine-tuned models)

**Cost Estimate:**
- Gemini API: ~$20-30 (400 tasks × 5 gen × $0.01/call)
- Kaggle GPU: Free (within weekly quota)
- **Total:** $20-30

---

## Documentation Updates

After completion, update:
- `README.md` - Session Handover with Phase 4b results
- `docs/phase4b_distillation_results.md` - Detailed analysis
- `CLAUDE.md` - Next priority tasks
- `~/.claude/core-patterns.md` - Knowledge distillation pattern

---

**Status:** Ready for execution pending Task 4.4 completion
**Owner:** TBD (awaiting user baseline submission)
**Priority:** CRITICAL (needed for competitive score)

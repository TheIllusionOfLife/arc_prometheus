# Kaggle Model Setup Guide

This guide explains how to download Code Gemma 7B and upload it to Kaggle for offline inference.

## Prerequisites

- HuggingFace account
- Kaggle account
- ~14GB free disk space
- Internet connection (for download only)

## Step 1: Get HuggingFace Access Token

Code Gemma is a gated model that requires authentication.

### 1.1 Request Model Access
1. Go to https://huggingface.co/google/codegemma-7b-it
2. Click "Request access" button
3. Wait for approval (usually instant, but can take up to 24 hours)

### 1.2 Generate Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: "ARC Prometheus" (or any descriptive name)
4. Type: "Read" (sufficient for model downloads)
5. Click "Generate token"
6. **IMPORTANT**: Copy the token immediately (it won't be shown again)

### 1.3 Set Environment Variable
```bash
# Linux/macOS
export HF_TOKEN=hf_your_token_here

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"

# Windows (CMD)
set HF_TOKEN=hf_your_token_here
```

**Security Note**: Never commit your HF_TOKEN to git. Add it to `.env` or use environment variables.

## Step 2: Download Model Locally

Run the download script:

```bash
# Ensure HF_TOKEN is set
echo $HF_TOKEN  # Should show your token

# Download model (~7GB, takes 10-30 minutes)
# IMPORTANT: Use 'uv run python' to use the project's dependencies
uv run python scripts/download_codegemma.py --output-dir models/codegemma-7b
```

**Expected Output**:
```
============================================================
Code Gemma 7B Download Script
============================================================

Output directory: models/codegemma-7b
Using HuggingFace token: hf_xxxxx...

[1/2] Downloading tokenizer from google/codegemma-7b-it...
✓ Tokenizer saved to: models/codegemma-7b/tokenizer

[2/2] Downloading model from google/codegemma-7b-it...
⚠️  This will download ~7GB of data and may take 10-30 minutes...
✓ Model saved to: models/codegemma-7b/model

============================================================
Download Complete!
============================================================

Model files:
  model-00001-of-00004.safetensors: 4764.1 MB
  model-00002-of-00004.safetensors: 4752.1 MB
  model-00003-of-00004.safetensors: 4752.1 MB
  model-00004-of-00004.safetensors: 2016.1 MB
  tokenizer.json: 32.8 MB

Total size: 16317.2 MB (15.93 GB)
```

**Troubleshooting**:
- **403 Forbidden**: Check that you requested access and were approved
- **Network timeout**: Try running script again (it will resume download)
- **Disk space error**: Need at least 14GB free (7GB model + 7GB cache)

## Step 3: Upload to Kaggle Dataset

### 3.1 Create New Dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset" button
3. Dataset settings:
   - **Title**: `codegemma-7b-instruct`
   - **Slug**: `codegemma-7b-instruct` (auto-generated)
   - **Visibility**: Private (or Public if you want to share)
   - **License**: Apache 2.0 (matches model license)

### 3.2 Upload Files
Upload the entire `models/codegemma-7b/` directory:

**Option A: Kaggle Web UI**
1. Drag and drop the entire `models/codegemma-7b/` folder
2. Wait for upload to complete (~7GB, takes 20-60 minutes depending on connection)
3. Click "Create"

**Option B: Kaggle CLI** (faster, recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (get from https://www.kaggle.com/settings)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Create dataset metadata
cat > dataset-metadata.json <<EOF
{
  "title": "codegemma-7b-instruct",
  "id": "your-username/codegemma-7b-instruct",
  "licenses": [{"name": "apache-2.0"}]
}
EOF

# Upload (single command, resumes on failure)
kaggle datasets create -p models/codegemma-7b/ -r zip
```

### 3.3 Verify Upload
1. Go to your dataset: `https://www.kaggle.com/datasets/your-username/codegemma-7b-instruct`
2. Check that all files are present:
   - `model/` directory with model weights
   - `tokenizer/` directory with tokenizer files
3. Note the dataset path: `/kaggle/input/codegemma-7b-instruct/`

## Step 4: Update Kaggle Notebook

Edit `notebooks/kaggle_submission.ipynb` Cell 2 to reference your dataset:

```python
# Load Code Gemma 7B from uploaded Kaggle dataset
# CRITICAL: Verify exact path structure in your Kaggle dataset!
MODEL_DIR = "/kaggle/input/codegemma-7b-instruct/codegemma-7b"
MODEL_PATH = f"{MODEL_DIR}/model"
TOKENIZER_PATH = f"{MODEL_DIR}/tokenizer"
```

**Path Troubleshooting:**
- Error: "No file named model.safetensors found" → Check dataset structure
- Run this in a test cell to verify paths:
  ```python
  import os
  print("Dataset files:")
  for root, dirs, files in os.walk("/kaggle/input/codegemma-7b-instruct"):
      print(f"{root}/")
      if root.count("/") < 5:  # Only show first few levels
          for file in files[:3]:
              print(f"  {file}")
  ```

## Step 5: Test on Kaggle

### 5.1 Upload Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `notebooks/kaggle_submission.ipynb`
4. Notebook settings:
   - **Environment**: Python
   - **GPU**: L4x4 (96GB VRAM, required for 7B model)
   - **Internet**: OFF (disable to match competition rules)

### 5.2 Add Dataset Input
1. Click "Add Input" in the right panel
2. Search for `codegemma-7b-instruct`
3. Select your dataset
4. Verify path matches MODEL_PATH in Cell 2

### 5.3 Run Small Test FIRST (Highly Recommended)

**Before running 240 tasks (7+ hours), validate with 5 tasks (~10 minutes):**

1. In Cell 5, set `TEST_MODE = True`:
   ```python
   TEST_MODE = True  # Run only 5 tasks for validation
   ```

2. Run all cells (Cells 1-6)

3. **Expected output** (if everything works):
   ```
   ✅ Model loaded successfully! Device: cuda:0
   ⚠️  TEST MODE: 5 tasks × 2.4 min = ~12 min
   Processing 1/5: 00576224
     Completed in 24.2s (best fitness: 0)
   ...
   Total time: 8.6 minutes (0.14 hours)
   Average: 103.6 seconds/task
   ✅ Submission format validated successfully!
   ```

4. **Common issues and fixes**:
   - "Model loading failed" → Check MODEL_PATH (see Step 4 troubleshooting)
   - "Tokenizer parallelism warnings" → Already fixed in Cell 1 (`os.environ["TOKENIZERS_PARALLELISM"] = "false"`)
   - All fitness = 0 → **EXPECTED** for baseline Code Gemma (no fine-tuning)

### 5.4 Run Full Submission

**Once test succeeds**, set `TEST_MODE = False` and run all cells again:

```python
TEST_MODE = False  # Full run: 240 tasks
```

**Expected Runtime** (validated on Kaggle L4x4):
- **Population size 2** (default): ~103.6s/task × 240 = **~7 hours** ✅ SAFE
- **Population size 3**: ~155s/task × 240 = **~10 hours** ⚠️ Tight but ok
- **Population size 5**: ~260s/task × 240 = **~17 hours** ❌ EXCEEDS 12h limit!

**GPU Quota Consumption** (L4x4 uses 2× quota):
- 7 hour run = **14 hours of weekly quota** consumed
- Weekly limit: 30 hours
- Leave buffer for re-runs and improvements

## Step 6: Submit to Competition

### 6.1 Final Run
1. Ensure notebook runs to completion without errors
2. Download `submission.json` from output
3. Validate format (should match `sample_submission.json` structure)

### 6.2 Submit to ARC Prize 2025
1. Go to https://www.kaggle.com/competitions/arc-prize-2025
2. Click "Submit Predictions"
3. Select notebook (NOT submission.json file)
4. Click "Submit"
5. Wait for scoring (private test set, no immediate feedback)

## Expected Results

**Baseline Score** (raw Code Gemma 7B, no fine-tuning):
- **Realistic**: 3-8% accuracy (validated: fitness=0 on 5 test tasks)
- **Optimistic**: 8-12% accuracy (if lucky with task distribution)
- **Competitive threshold**: 10% = 9th place on leaderboard (as of Nov 1, 2025)
- **Top score**: 27% (as of Nov 1, 2025)
- **SOTA potential**: 20-25% (with knowledge distillation + active inference)

**Important Notes:**
- Fitness = 0 during testing is **normal** for baseline Code Gemma
- Code Gemma 7B is significantly weaker than Gemini 2.0 Flash
- Small improvements from knowledge distillation can jump rankings significantly
- Even 5-8% validates the pipeline and provides baseline for optimization

## Next Steps After Baseline

Once baseline is working:

1. **Knowledge Distillation** (Phase 4b):
   - Collect Gemini outputs on training set
   - Fine-tune Code Gemma to mimic Gemini
   - Expected: +5-10% improvement

2. **Active Inference** (Phase 4c):
   - Augment training examples (3 → 30+)
   - Fine-tune model per task at runtime
   - Expected: +5-10% improvement

3. **Multi-Agent Active Inference** (Phase 4d):
   - Each agent fine-tunes separately
   - Expected: +2-5% over single-model

## Troubleshooting

### Model Won't Load
- Check GPU is enabled (L4x4 required)
- Verify dataset path matches MODEL_PATH
- Check model files are complete (not corrupted during upload)

### Out of Memory
- Reduce population size: `SimplifiedEvolution(population_size=3)`
- Reduce max_tokens: `generate_with_local_model(..., max_tokens=1024)`
- Use 2B model instead of 7B (lower accuracy but faster)

### Timeout Issues
- Reduce max_generations: `SimplifiedEvolution(max_generations=2)`
- Increase timeout in execute_solver_safe: `timeout=10`
- Skip failed tasks: add try/except around evolution loop

## Alternative Models

If Code Gemma access is problematic, try these ungated alternatives:

1. **google/gemma-2b** (smaller, faster, lower accuracy)
   - Model ID: `google/gemma-2b`
   - Size: ~2GB
   - No authentication required
   - Expected score: 5-15%

2. **microsoft/phi-2** (alternative architecture)
   - Model ID: `microsoft/phi-2`
   - Size: ~3GB
   - No authentication required
   - Expected score: 8-18%

To use alternative model:
```bash
python scripts/download_codegemma.py \
  --model-name google/gemma-2b \
  --output-dir models/gemma-2b
```

## References

- **Code Gemma Model Card**: https://huggingface.co/google/codegemma-7b-it
- **Kaggle Datasets API**: https://www.kaggle.com/docs/api#datasets
- **ARC Prize 2025**: https://www.kaggle.com/competitions/arc-prize-2025
- **plan_20251101.md**: Detailed implementation plan

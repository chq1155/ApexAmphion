# Amphion RL Toolkit

Amphion is an end-to-end pipeline for **antimicrobial peptide (AMP)** sequence generation.  
The repository bundles supervised fine-tuning (SFT), reward-model training, PPO-based
reinforcement learning, and sampling utilities that work on top of the ProGen language model.

The PPO reward used for reinforcement learning now blends the classification head
(`reward_amp_cls`) and the physicochemical property heuristic (`prop_reward`) **1:1** via
`CompositeReward`, ensuring both biological plausibility and desired chemical profiles.

## Repository Layout

- `amphion_sft/` – LoRA-based supervised fine-tuning utilities for ProGen.
- `amphion_rl/` – PPO training loop, reward definitions, and sampling helpers.
- `cls_reward/` – MLP classifier and data processing for AMP reward modeling.
- `data_processing/` – Tools for cleaning and splitting raw AMP datasets.
- `checkpoint/` – Expected location for base models and stage outputs.
- `train.py` – Orchestrates SFT, classifier, and PPO training stages.
- `generate.py` – High-level CLI for sampling sequences from PPO checkpoints.
- `environment.yml` – Conda environment with CPU/GPU and Macrel dependencies.

## Requirements

- Linux with CUDA-capable GPU (recommended ≥24 GB for PPO).
- Conda or Mamba.
- Access to the ProGen weights and other checkpoints (user-provided via shared link).

Create the environment:

```bash
conda env create -f environment.yml
conda activate ppo
# (optional) install the repo in editable mode
pip install -e .
```

## Data & Checkpoint Layout

All paths default to the repository root. You can override them by setting `AMPGEN_HOME`
before running any scripts (see `env.py` for details).

```
ApexAmphion/
├─ data/
│  ├─ sft_train.csv
│  ├─ sft_valid.csv
│  ├─ cls_reward_train.csv
│  ├─ cls_reward_valid.csv
│  ├─ cls_reward_train.pkl        # auto-generated if missing
│  └─ cls_reward_valid.pkl        # auto-generated if missing
├─ checkpoint/
│  ├─ progen2-xlarge/             # base ProGen weights + tokenizer
│  ├─ amphion-sft/                # SFT LoRA weights (output of stage 1)
│  ├─ amphion-rl/                 # PPO LoRA weights (output of stage 3)
│  └─ apexmic/
│     └─ cls_reward.pth           # trained AMP classifier checkpoint
```

**Important:** Download the data and checkpoints from the provided drive and drop them
into the folders above before running any stage.

## Training Pipeline

`train.py` supports running the SFT, classifier, PPO stages individually or end-to-end.
All commands assume execution from the repository root with the `ppo` environment active.

### 1. Supervised Fine-Tuning (SFT)

```bash
python train.py --step sft \
  --sft_batch_size 16 \
  --sft_epochs 10 \
  --sft_path checkpoint/amphion-sft
```

The script reads `data/sft_{train,valid}.csv` and writes the best LoRA adaptor inside
`checkpoint/amphion-sft`.

### 2. Reward Classifier

```bash
python train.py --step cls_reward \
  --cls_reward_path checkpoint/apexmic/cls_reward.pth
```

If the `.pkl` ESM embeddings are missing, they will be generated automatically from
the CSV files. You can also preprocess manually:

```bash
python cls_reward/data_processing.py -i data/cls_reward_train.csv
python cls_reward/data_processing.py -i data/cls_reward_valid.csv
```

### 3. PPO with Composite Reward

```bash
python train.py --step ppo \
  --ppo_epochs 2 \
  --ppo_batch_size 128 \
  --ppo_mini_batch_size 32 \
  --ppo_lr 1e-5 \
  --ppo_path checkpoint/amphion-rl \
  --cls_reward_path checkpoint/apexmic/cls_reward.pth \
  --sft_path checkpoint/amphion-sft
```

The PPO loop automatically combines `reward_amp_cls` and `prop_reward` in a 50/50 ratio.
Training stats (including component-wise reward means) are logged to Weights & Biases;
set `WANDB_MODE=disabled` to keep logs local.

To run all three stages sequentially:

```bash
python train.py --step all
```

## Generation & Evaluation

Sample peptides from a PPO checkpoint:

```bash
python generate.py \
  --ppo checkpoint/amphion-rl \
  --cls_reward checkpoint/apexmic/cls_reward.pth \
  --mode cls_prop \
  --batch 32 \
  --total 3200
```

The script rewrites the PPO adaptor metadata to point at `checkpoint/progen2-xlarge`,
loads the composite reward, and prints the filtered top candidates. Adjust `--threshold`
and `--top_n` to tighten selection criteria.

You can also call the lower-level helper directly:

```python
from amphion_rl import generate
tokenizer, model = generate.load_trl("checkpoint/amphion-rl")
reward = generate.CompositeReward(...)
sequences, scores = generate.sampling_trl(tokenizer, model, reward)
```

## Testing & Validation

- `python -m compileall amphion_rl amphion_sft` verifies that the Python sources compile.
- Add unit tests for new functionality under `tests/` (not yet provided) and run with `pytest`.
- `test.py` demonstrates the progress-bar utilities used during training.

## Notes & Best Practices

- Ensure `macrel` binaries from the Conda environment are on your `PATH`; the property
  reward runs Macrel in batch mode.
- Check GPU utilization during PPO — the ESM encoder and classifier live on the same device.
- Maintain consistent random seeds during experimentation via the `--seed` flag.

## License

Pending; please choose an open-source license before publishing the project publicly.

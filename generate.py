import argparse
import json
import os
from functools import partial

import torch

from amphion_rl import generate as rl_generate
from amphion_rl.reward import CompositeReward, reward_amp_cls, prop_reward
from amphion_rl.utils import loading_esm
from cls_reward.mlp import MLP
from env import MODEL_PATH, PROGEN_PATH


def build_reward_function(mode: str, cls_checkpoint: str, device: torch.device):
    """Create the reward callable matching the requested mode."""
    if mode in {"cls", "cls_prop"}:
        batch_converter, esm_model, alphabet = loading_esm()
        esm_model = esm_model.to(device)
        cls_model = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
        cls_model.load_state_dict(torch.load(cls_checkpoint, weights_only=True))

        if mode == "cls_prop":
            return CompositeReward(
                esm_model=esm_model,
                batch_converter=batch_converter,
                alphabet=alphabet,
                cls_model=cls_model,
                cls_weight=0.5,
                prop_weight=0.5,
                device=str(device),
            )

        return partial(
            reward_amp_cls,
            esm_model=esm_model,
            batch_converter=batch_converter,
            alphabet=alphabet,
            cls1=cls_model,
            device=str(device),
        )

    if mode == "prop":
        return prop_reward

    raise ValueError(f"Unsupported reward mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Sample sequences from a PPO adaptor.")
    parser.add_argument("--seed", type=int, default=0, help="Set to >0 for determinism.")
    parser.add_argument(
        "--ppo",
        default=os.path.join(MODEL_PATH, "ppo"),
        help="Directory containing the PPO LoRA adaptor to load.",
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Number of sequences sampled per batch."
    )
    parser.add_argument(
        "--total", type=int, default=3200, help="Total number of samples to draw."
    )
    parser.add_argument(
        "--cls_reward",
        type=str,
        default=os.path.join(MODEL_PATH, "cls_reward.pth"),
        help="Checkpoint of the AMP classifier.",
    )
    parser.add_argument(
        "--mode",
        choices=["cls", "prop", "cls_prop"],
        default="cls_prop",
        help="Reward shaping strategy to evaluate generated sequences.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Filter threshold passed to `filter_sequences`.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=60,
        help="Number of top sequences to retain after filtering.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.synchronize()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"

    if args.seed:
        rl_generate.set_seed(args.seed)

    adapter_config = os.path.join(args.ppo, "adapter_config.json")
    if os.path.exists(adapter_config):
        with open(adapter_config, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["base_model_name_or_path"] = PROGEN_PATH
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    tokenizer, trl_progen = rl_generate.load_trl(path=args.ppo)
    reward_func = build_reward_function(args.mode, args.cls_reward, device)

    generated_seqs, reward_scores = rl_generate.sampling_trl(
        tokenizer,
        trl_progen,
        reward_func,
        batch=args.batch,
        total=args.total,
        device=device,
    )
    result, count = rl_generate.filter_sequences(
        generated_seqs, reward_scores, threshold=args.threshold, top_n=args.top_n
    )

    print(f"Filtered sequences: {result}")
    print(f"Total sequences under threshold: {count}")


if __name__ == "__main__":
    main()

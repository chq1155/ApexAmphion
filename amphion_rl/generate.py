import os
import random
import sys
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from cls_reward.mlp import MLP
from .reward import calculate_physchem_prop, prop_reward, reward_amp_cls, CompositeReward
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from .utils import loading_esm

from progen2hf_model import ProGenConfig, ProGenForCausalLM


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_trl(
    path: str,
) -> Tuple[AutoTokenizer, AutoModelForCausalLMWithValueHead]:
    """Load the TRL model and tokenizer."""
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True, torchscript=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    trl_progen = AutoModelForCausalLMWithValueHead.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    trl_progen.eval()

    return tokenizer, trl_progen


def filter_sequences(
    protein_sequences: List[str],
    scores: List[float],
    threshold: float = 2.0,
    top_n: int = 60,
) -> Tuple[Dict[str, float], int]:
    """Filter protein sequences based on scores and return top N results."""
    result = {
        protein: score
        for protein, score in zip(protein_sequences, scores)
        if score < threshold
    }
    sorted_items = sorted(result.items(), key=lambda x: x[1])[:top_n]

    for value, key in sorted_items:
        print(f"{value}: {key}")

    return dict(sorted_items), len(result)


@torch.no_grad()
def generate_sequences(
    model,
    tokenizer,
    generate_config: Dict,
    context: str,
    batch_size: int,
    device="cuda",
) -> Tuple[List[str], torch.Tensor]:
    """Generate sequences using the model."""
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, **generate_config)
    sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [s[:-1] for s in sequences], outputs


@torch.no_grad()
def sampling_trl(
    tokenizer,
    trl_progen,
    reward_func: Callable[[List[str]], torch.Tensor],
    batch: int = 32,
    total: int = 3200,
    device="cuda",
):
    """Sample sequences using the TRL model."""
    trl_progen = trl_progen.to(device)

    bad_words = ["B", "O", "U", "X", "Z"]
    generate_config = {
        "max_length": 51,
        "num_return_sequences": batch,
        "temperature": 1.0,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": [tokenizer.encode(word) for word in bad_words],
    }

    generate_func = partial(
        generate_sequences, trl_progen, tokenizer, generate_config, "<|bos|>"
    )

    generated_seqs_total = []
    reward_record = []

    for _ in tqdm(range(total // batch), desc="Generating sequences"):
        sequences, _ = generate_func(batch)
        generated_seqs_total.extend(sequences)

        rewards = reward_func(sequences).cpu().numpy().round(3)
        reward_record.extend(rewards)

    return generated_seqs_total, reward_record


if __name__ == "__main__":
    # Constants
    SEED = 3407
    MODEL_PATH = "/home/ubuntu/ApexAmphion/checkpoint/amphion-rl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer, trl_progen = load_trl(path=MODEL_PATH)
    batch = 32
    mode = "cls_prop"
    if mode in {"cls", "cls_prop"}:
        bad_words = ["B", "O", "U", "X", "Z"]
        batch_converter, esm_model, alphabet = loading_esm()
        esm_model = esm_model.to(device)
        cls1 = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
        cls1.load_state_dict(
            torch.load(
                "/home/ubuntu/ApexAmphion/checkpoint/apexmic/cls_reward.pth", weights_only=True
            ),
        )
        if mode == "cls_prop":
            reward_func = CompositeReward(
                esm_model=esm_model,
                batch_converter=batch_converter,
                alphabet=alphabet,
                cls_model=cls1,
                cls_weight=0.5,
                prop_weight=0.5,
                device=device,
            )
        else:
            reward_func = partial(
                reward_amp_cls,
                esm_model=esm_model,
                batch_converter=batch_converter,
                alphabet=alphabet,
                cls1=cls1,
            )
    elif mode == "prop":
        reward_func = prop_reward
    else:
        raise ValueError(f"Unsupported reward mode: {mode}")

    generated_seqs, reward_distances = sampling_trl(tokenizer, trl_progen, reward_func)
    result, count = filter_sequences(generated_seqs, reward_distances)

    print(f"Filtered sequences: {result}")
    print(f"Total sequences under threshold: {count}")

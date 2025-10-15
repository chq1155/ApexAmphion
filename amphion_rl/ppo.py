import os
import random
import wandb
import torch
import numpy as np
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm
from functools import partial

from cls_reward.mlp import MLP

from .utils import (
    clean_sequences,
    load_pretrained_progen_model,
    loading_esm,
)
from .data import loading_dataset
from .reward import reward_amp_cls, prop_reward, CompositeReward


def set_seed(seed):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Setting random seed as {seed}.")


def create_ppo_config(seed):
    return PPOConfig(
        tracker_project_name="ampgen_macrel",
        exp_name="ppo_stair",
        log_with="wandb",
        steps=500,
        learning_rate=1e-5,
        batch_size=128,
        mini_batch_size=32,
        gradient_accumulation_steps=4,
        ppo_epochs=2,
        early_stopping=True,
        is_peft_model=True,
        seed=seed,
        optimize_cuda_cache=True,
        optimize_device_cache=True,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
    )


def create_generate_config(tokenizer):
    bad_words = ["B", "O", "U", "X", "Z"]
    bad_words_ids = [tokenizer.encode(word) for word in bad_words]
    return {
        "max_length": 51,
        "num_return_sequences": 1,
        "temperature": 0.9,
        "num_beams": 4,
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": bad_words_ids,
    }


def log_stats(batch, reward_tensor, extra_logs=None):
    reward_tensor = reward_tensor.detach().cpu()
    logs = {
        "reward/max": reward_tensor.max().item(),
        "reward/mean": reward_tensor.mean().item(),
        "response/avg_length": sum(len(str(item)) for item in batch["response"])
        / len(batch["response"]),
    }
    if extra_logs:
        logs.update(extra_logs)
    wandb.log(logs)


def train(
    ppo_config,
    tokenizer,
    progen_model,
    generate_config,
    reward_func,
    output_path,
    cleanup_non_optimal=False,
):
    device = "cuda"

    ppo_dataloader = loading_dataset(ppo_config.steps, ppo_config.batch_size)

    trl_progen = AutoModelForCausalLMWithValueHead.from_pretrained(progen_model)

    ppo_trainer = PPOTrainer(model=trl_progen, config=ppo_config, tokenizer=tokenizer)

    step = 0

    best_score = None
    best_output = None

    for epoch in tqdm(range(ppo_config.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_dataloader):
            query_tensors = batch["input_ids"].squeeze(-1).to(device)
            query_tensors = [query_tensors[i] for i in range(len(query_tensors))]

            response_tensors = ppo_trainer.generate(query_tensors, **generate_config)

            batch["response"] = clean_sequences(
                [
                    tokenizer.decode(output_seq, skip_special_tokens=True)
                    for output_seq in response_tensors
                ]
            )
            print(batch["response"][:16])
            raw_reward = reward_func(batch["response"])
            if isinstance(raw_reward, torch.Tensor):
                reward_tensor = raw_reward.detach()
            elif isinstance(raw_reward, (list, tuple)):
                stacked = [
                    r.detach().reshape(-1)
                    if isinstance(r, torch.Tensor)
                    else torch.tensor([r], dtype=torch.float32)
                    for r in raw_reward
                ]
                reward_tensor = torch.cat(stacked).to(device=device, dtype=torch.float32)
            else:
                reward_tensor = torch.tensor(raw_reward, device=device, dtype=torch.float32)

            component_logs = {}
            last_components = getattr(reward_func, "last_components", None)
            if last_components:
                component_logs = {
                    "reward/cls_raw_mean": last_components["cls_raw"].mean().item(),
                    "reward/prop_raw_mean": last_components["prop_raw"].mean().item(),
                    "reward/cls_norm_mean": last_components["cls_norm"].mean().item(),
                    "reward/prop_norm_mean": last_components["prop_norm"].mean().item(),
                }

            log_stats(batch, reward_tensor, component_logs)

            reward_score = [r.reshape(-1) for r in reward_tensor]

            stats = ppo_trainer.step(query_tensors, response_tensors, reward_score)
            ppo_trainer.log_stats(
                stats, batch, reward_score, columns_to_log=["input_ids", "response"]
            )
            step += 1

            if step % 50 == 0:
                mean_score = reward_tensor.mean().item()
                if best_score is None or mean_score > best_score:
                    best_score = mean_score
                    if cleanup_non_optimal and best_output is not None:
                        os.rmdir(best_output)
                    save_path = os.path.join(output_path, f"{seed}_{step}")
                    ppo_trainer.save_pretrained(save_path)
                    best_output = save_path
    return best_output, best_score


if __name__ == "__main__":
    seed = 822
    mode = "cls_prop"  # options: 'cls', 'prop', 'cls_prop'
    set_seed(seed)
    device = "cuda"
    ppo_config = create_ppo_config(seed)

    tokenizer, progen_model = load_pretrained_progen_model(
        pretrain_path="/root/ubuntu/ApexAmphion/checkpoint/amphion-sft",
        protein_tokenizer_path="/root/ubuntu/ApexAmphion/checkpoint/progen2-xlarge",
    )
    tokenizer.pad_token = tokenizer.eos_token
    generate_config = create_generate_config(tokenizer)

    print(ppo_config)
    print(generate_config)

    if mode in {"cls", "cls_prop"}:
        batch_converter, esm_model, alphabet = loading_esm()
        esm_model = esm_model.to(device)
        cls1 = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
        cls1.load_state_dict(
            torch.load(
                "/root/ubuntu/ApexAmphion/checkpoint/apexmic/cls_reward.pth",
                weights_only=True,
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

        def reward_func_prop(seqs):
            reward_score = prop_reward(seqs)
            reward_score = (reward_score - 0.5) * 2.0
            return reward_score

        reward_func = reward_func_prop

    else:
        raise ValueError(f"Unsupported reward mode: {mode}")

    train(
        ppo_config=ppo_config,
        tokenizer=tokenizer,
        progen_model=progen_model,
        generate_config=generate_config,
        reward_func=reward_func,
        output_path="/home/ubuntu/ApexAmphion/checkpoint/amphion-rl",
    )

import argparse
import csv
from enum import Enum
import os
from functools import partial
import random

import numpy as np
import torch
import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import wandb
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


from amphion_rl.reward import reward_amp_cls, CompositeReward
from amphion_rl.utils import load_pretrained_progen_model, loading_esm
import amphion_sft.sft_ampgen as sft_train
from cls_reward import data_processing
from cls_reward import train as reward_train
from cls_reward.mlp import MLP, FocalLoss
from env import DATA_PATH, PROGEN_PATH, MODEL_PATH
from amphion_rl import ppo


def train_sft(batch_size, num_epochs, output_path):
    lr = 1e-4
    lora_r = 32
    lora_alpha = 32
    device = "cuda"
    train_path = os.path.join(DATA_PATH, "sft_train.csv")
    valid_path = os.path.join(DATA_PATH, "sft_valid.csv")
    protein_tokenizer_path = PROGEN_PATH
    protein_lang_encoder_path = PROGEN_PATH
    use_local_files = True

    protein_seq_tokenizer = AutoTokenizer.from_pretrained(
        protein_tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        protein_lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        torchscript=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["out_proj", "qkv_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_data = list(csv.reader(open(train_path)))[:20000]
    val_data = list(csv.reader(open(valid_path)))
    sampler_train = RandomSampler(train_data)
    sampler_val = RandomSampler(val_data)

    collate_fn = partial(sft_train.collate, tokenizer=protein_seq_tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=sampler_train,
        collate_fn=collate_fn,
        batch_size=batch_size,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, sampler=sampler_val, collate_fn=collate_fn, batch_size=16
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=(len(train_loader) * num_epochs) * 0.1,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    output, loss = sft_train.train(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        output_path=MODEL_PATH,
        cleanup_non_optimal=True,
        device=device,
    )
    os.rename(output, output_path)


def train_cls_reward(output_path):
    device = "cuda"
    train_path = os.path.join(DATA_PATH, "cls_reward_train.pkl")
    valid_path = os.path.join(DATA_PATH, "cls_reward_valid.pkl")

    # process data
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        model, batch_converter = data_processing.load_model()
        if not os.path.exists(train_path):
            csv_path = os.path.join(DATA_PATH, "cls_reward_train.csv")
            data_processing.process_csv(csv_path, train_path, model, batch_converter)
        if not os.path.exists(valid_path):
            csv_path = os.path.join(DATA_PATH, "cls_reward_valid.csv")
            data_processing.process_csv(csv_path, valid_path, model, batch_converter)

    # train
    train_dataloader = reward_train.get_dataloader(train_path, batch_size=32)
    valid_dataloader = reward_train.get_dataloader(valid_path, batch_size=32)

    model = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
    criterion = FocalLoss(alpha=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    reward_train.train_model(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        device,
        num_epochs=25,
        save_path=output_path,
    )


def train_ppo(
    model_path,
    progen_path,
    cls_reward_path,
    seed,
    epochs,
    batch_size,
    mini_batch_size,
    lr,
    temperature,
    num_beams,
    repetition_penalty,
    top_p,
    length_penalty,
    output_path,
):
    device = "cuda"

    ppo_config = PPOConfig(
        tracker_project_name="ampgen_macrel",
        exp_name="ppo_stair",
        log_with="wandb",
        steps=500,
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=batch_size // mini_batch_size,
        ppo_epochs=epochs,
        early_stopping=True,
        is_peft_model=True,
        seed=seed,
        optimize_cuda_cache=True,
        optimize_device_cache=True,
        use_score_scaling=True,
        use_score_norm=True,
        whiten_rewards=True,
    )
    tokenizer, progen_model = load_pretrained_progen_model(
        pretrain_path=model_path,
        protein_tokenizer_path=progen_path,
    )
    tokenizer.pad_token = tokenizer.eos_token
    bad_words = ["B", "O", "U", "X", "Z"]
    bad_words_ids = [tokenizer.encode(word) for word in bad_words]
    generate_config = {
        "max_length": 51,
        "num_return_sequences": 1,
        "temperature": temperature,
        "num_beams": num_beams,
        "top_p": top_p,
        "do_sample": True,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": bad_words_ids,
    }

    # Constants
    batch_converter, esm_model, alphabet = loading_esm()
    esm_model = esm_model.to(device)
    cls1 = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
    cls1.load_state_dict(
        torch.load(
            cls_reward_path,
            weights_only=True,
        ),
    )
    reward_func = CompositeReward(
        esm_model=esm_model,
        batch_converter=batch_converter,
        alphabet=alphabet,
        cls_model=cls1,
        cls_weight=0.5,
        prop_weight=0.5,
        device=device,
    )

    output, score = ppo.train(
        ppo_config=ppo_config,
        tokenizer=tokenizer,
        progen_model=progen_model,
        generate_config=generate_config,
        reward_func=reward_func,
        output_path=MODEL_PATH,
        cleanup_non_optimal=True,
    )
    os.rename(output, output_path)


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self


class Step(Enum):
    sft = "sft"
    cls_reward = "cls_reward"
    ppo = "ppo"
    all = "all"

    def __str__(self):
        return self.value


parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("-s", "--step", type=Step, default=Step.all)
parser.add_argument("--seed", type=int, default=0)

# sft args
parser.add_argument("--sft_batch_size", type=int, default=16)
parser.add_argument("--sft_epochs", type=int, default=10)
parser.add_argument("--sft_path", default=os.path.join(MODEL_PATH, "sft"))

# cls_reward args
parser.add_argument(
    "--cls_reward_path", default=os.path.join(MODEL_PATH, "cls_reward.pth")
)


# ppo args
parser.add_argument("--ppo_epochs", type=int, default=2)
parser.add_argument("--ppo_batch_size", type=int, default=128)
parser.add_argument("--ppo_mini_batch_size", type=int, default=32)
parser.add_argument("--ppo_lr", type=float, default=1e-5)
parser.add_argument(
    "--ppo_temperature", type=float, default=0.9, choices=Range(0.7, 1.2)
)
parser.add_argument("--ppo_beams", type=int, default=4, choices=[1, 2, 4, 8])
parser.add_argument(
    "--ppo_repetition_penalty", type=float, default=1.2, choices=Range(1.0, 1.4)
)
parser.add_argument("--ppo_top_p", type=float, default=0.95, choices=[0.95, 1.0])
parser.add_argument(
    "--ppo_length_penalty", type=float, default=1.0, choices=Range(0.8, 1.2)
)
parser.add_argument("--ppo_path", default=os.path.join(MODEL_PATH, "ppo"))


if __name__ == "__main__":
    torch.cuda.synchronize()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        project="AMPGen_AMPSphere",
        config={},
    )

    args = parser.parse_args()

    if args.seed != 0:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Setting random seed as {seed}.")

    if args.step == Step.all or args.step == Step.sft:
        train_sft(
            batch_size=args.sft_batch_size,
            num_epochs=args.sft_epochs,
            output_path=args.sft_path,
        )
    if args.step == Step.all or args.step == Step.cls_reward:
        train_cls_reward(output_path=args.cls_reward_path)
    if args.step == Step.all or args.step == Step.ppo:
        train_ppo(
            model_path=args.sft_path,
            progen_path=PROGEN_PATH,
            cls_reward_path=args.cls_reward_path,
            seed=args.seed,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            mini_batch_size=args.ppo_mini_batch_size,
            lr=args.ppo_lr,
            temperature=args.ppo_temperature,
            num_beams=args.ppo_beams,
            repetition_penalty=args.ppo_repetition_penalty,
            top_p=args.ppo_top_p,
            length_penalty=args.ppo_length_penalty,
            output_path=args.ppo_path,
        )

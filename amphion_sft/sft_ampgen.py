import csv
import os
import random
from functools import partial

import numpy as np
import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from progen2hf_model import ProGenConfig, ProGenForCausalLM

AutoConfig.register("progen", ProGenConfig)
AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)


def eval_one_epoch(data_loader, model, device="cuda"):
    model.eval()
    eval_loss = 0.0
    print("--------------------------------")
    print("Start Evaluation")
    print("--------------------------------")
    with torch.no_grad():
        vc = 0
        for step, batch in enumerate(tqdm(data_loader)):
            vc += 1
            input_ids = batch["input_ids"].to(device)
            masks = batch["masks"].to(device)

            input_ = input_ids
            masks_ = masks.clone()

            label_ = input_.clone()
            label_[:, 0] = -100
            label_mask = input_ == 0
            label_[label_mask] = -100

            input_emb = model.transformer.wte(input_)
            outputs = model(
                inputs_embeds=input_emb, attention_mask=masks_, labels=label_
            )
            loss = outputs[0]

            eval_loss += loss.detach().float()

        print(eval_loss / vc)
        print(f"Evaluation Loss: {eval_loss/vc}")
        wandb.log({"Valid Loss": eval_loss / vc})

    return eval_loss / vc


def train(
    model,
    num_epochs,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    output_path="",
    cleanup_non_optimal=False,
    device="cuda",
):
    model = model.to(device)
    best_loss = None
    best_output = None
    for epoch in range(num_epochs):
        print("--------------------------------")
        print("Start Training")
        print("--------------------------------")
        model.train()
        total_loss = 0.0
        tc = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=True)
        for step, batch in enumerate(progress_bar):
            tc += 1
            input_ids = batch["input_ids"].to(device)
            masks = batch["masks"].to(device)

            input_ = input_ids
            masks_ = masks.clone()

            label_ = input_.clone()
            label_[:, 0] = -100
            label_mask = input_ == 0
            label_[label_mask] = -100

            input_emb = model.transformer.wte(input_)
            outputs = model(
                inputs_embeds=input_emb, attention_mask=masks_, labels=label_
            )
            loss = outputs[0]
            wandb.log({"Train Step Loss": loss})

            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        eval_loss_epoch = eval_one_epoch(val_loader, model)
        save_path = os.path.join(output_path, f"sft_{epoch}_{tc}_{eval_loss_epoch}")

        if best_loss is None or eval_loss_epoch < best_loss:
            if cleanup_non_optimal and best_output is not None:
                os.rmdir(best_output)
            best_loss = eval_loss_epoch
            best_output = save_path
            model.save_pretrained(best_output)

        model.train()
        print(f"Training Loss Epoch{epoch+1}: {total_loss/tc}")
    return best_output, best_loss


def collate(batch, tokenizer):

    input_ids = []
    masks = []

    for (
        name,
        sequence,
    ) in batch:

        encoded = tokenizer(sequence, truncation=True, max_length=50)["input_ids"]
        encoded = torch.tensor(encoded, dtype=torch.int64)
        end_token = torch.zeros(1, dtype=torch.int64) + 2
        encoded = torch.cat([encoded, end_token])
        pad = torch.zeros(51 - encoded.shape[0], dtype=torch.int64)
        padded = torch.cat([encoded, pad])
        start_token = torch.zeros(1, dtype=torch.int64) + 1
        padded = torch.cat([start_token, padded]).requires_grad_(False)

        mask = torch.zeros_like(padded, dtype=torch.float32)
        mask[padded != 0] = 1
        mask = mask.requires_grad_(False)

        input_ids.append(padded.clone().detach())
        masks.append(mask.clone().detach())

    return {
        "input_ids": torch.stack(input_ids),
        "masks": torch.stack(masks),
    }


if __name__ == "__main__":
    torch.cuda.synchronize()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Setting random seed as {seed}.")

    lr = 1e-4
    num_epochs = 10
    lora_r = 32
    lora_alpha = 32
    device = "cuda"
    train_path = "/home/ubuntu/data/v1_data/all_finetune_train.csv"
    valid_path = "/home/ubuntu/data/v1_data/all_finetune_valid.csv"
    protein_tokenizer_path = "/home/ubuntu/progen2-xlarge"
    protein_lang_encoder_path = "/home/ubuntu/progen2-xlarge"
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

    collate_fn = partial(collate, tokenizer=protein_seq_tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=sampler_train,
        collate_fn=collate_fn,
        batch_size=16,
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

    wandb.init(
        project="AMPGen_AMPSphere",
        config={"lora_r": lora_r, "lora_alpha": lora_alpha, "lr": lr},
    )

    train(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

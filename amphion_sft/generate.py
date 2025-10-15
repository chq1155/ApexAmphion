import re
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from progen2hf_model import ProGenConfig, ProGenForCausalLM


def setup_model_and_tokenizer(tokenizer_path, model_path, use_local_files=True):
    """Set up the model and tokenizer."""
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        torchscript=True,
    )

    return model, tokenizer


def apply_peft_and_load_weights(model, peft_config_params, pretrain_path):
    """Apply PEFT to the model and load pre-trained weights."""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=True, **peft_config_params
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading Pre-trained Model...")
    model.load_state_dict(torch.load(pretrain_path, map_location="cpu"))
    print("Successfully Loaded Pre-trained Weights!")

    return model


def generate_sequences(model, tokenizer, context, generation_params):
    """Generate sequences using the model."""
    print("Starting Generation...")
    with torch.no_grad():
        model = model.to(generation_params["device"])
        input_ids = (
            torch.tensor(tokenizer.encode(context))
            .view([1, -1])
            .to(generation_params["device"])
        )
        tokens_batch = model.base_model.model.generate(
            input_ids, do_sample=True, **generation_params
        )
        as_lists = lambda batch: [
            batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])
        ]
        return tokenizer.batch_decode(as_lists(tokens_batch))


def clean_sequences(generated_seq):
    """Clean the generated sequences."""
    clean_list = []
    exceed_count = 0
    for text in generated_seq:
        if "<|eos|>" not in text:
            exceed_count += 1
        text = re.sub(r"<\|bos\|>|<\|amp\|>|<\|pad\|>|<\|eos\|>", "", text)
        clean_list.append(text)
    print(
        f"Total number of exceed sequences: {exceed_count} out of {len(generated_seq)}."
    )
    return clean_list


def write_to_file(sequences, filename):
    """Write sequences to a file."""
    with open(filename, "w") as file:
        for seq in sequences:
            file.write(f"{seq}\n")


def main():
    # Configuration
    tokenizer_path = "/home/ubuntu/ApexAmphion/checkpoint/progen2-xlarge"
    model_path = "/home/ubuntu/ApexAmphion/checkpoint/progen2-xlarge"
    pretrain_path = "/home/ubuntu/ApexAmphion/checkpoint/progen2-xlarge/13_0_159.pt"

    peft_config_params = {
        "r": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["qkv_proj", "out_proj"],
    }

    generation_params = {
        "device": "cuda",
        "temperature": 0.5,
        "max_length": 53,
        "top_p": 0.95,
        "num_return_sequences": 100,
        "pad_token_id": 0,
    }

    # Setup and generate
    model, tokenizer = setup_model_and_tokenizer(tokenizer_path, model_path)
    model = apply_peft_and_load_weights(model, peft_config_params, pretrain_path)
    model.eval()

    generated_seq = generate_sequences(model, tokenizer, "<|bos|>", generation_params)
    clean_list = clean_sequences(generated_seq)

    # Write to file
    filename = (
        f'sample_{generation_params["temperature"]}_{generation_params["top_p"]}.txt'
    )
    write_to_file(clean_list, filename)


if __name__ == "__main__":
    main()

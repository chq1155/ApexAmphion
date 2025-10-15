import re
import esm
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from progen2hf_model import ProGenConfig, ProGenForCausalLM


def rename_state_dict(state_dict):
    return {
        (
            key.replace(".weight", ".base_layer.weight")
            if key.startswith("base_model.model.transformer.h.")
            and (
                key.endswith(".attn.qkv_proj.weight")
                or key.endswith(".attn.out_proj.weight")
            )
            else key
        ): value
        for key, value in state_dict.items()
    }


def load_progen_model(
    model_path="/home/ubuntu/progen2-xlarge",
    pretrain_path="/uac/gds/hqcao23/hqcao/ampgen_new/ppo_amp/9_600_138.pt",
    inference_mode=False,
):

    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, torchscript=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=inference_mode,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["qkv_proj", "out_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if not inference_mode:
        state_dict = torch.load(pretrain_path, map_location="cpu")
        model.load_state_dict(rename_state_dict(state_dict))
        print("Successfully Loaded Pre-trained Peft-Model Weights!")

    model.eval()
    return tokenizer, model


def load_pretrained_progen_model(
    pretrain_path: str,
    protein_tokenizer_path="/home/ubuntu/progen2-xlarge",
    inference_mode=False,
):
    AutoConfig.register("progen", ProGenConfig)
    AutoModelForCausalLM.register(ProGenConfig, ProGenForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(
        protein_tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_path,
        local_files_only=True,
        trust_remote_code=True,
        torchscript=False,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=inference_mode,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["qkv_proj", "out_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.eval()
    return tokenizer, model


def refine_sequences(sequences):
    patterns = ["<|bos|>", "<|amp|>", "<|pad|>", "<|eos|>"]
    return [re.sub("|".join(patterns), "", seq) for seq in sequences]


def clean_sequences(sequences):
    return ["".join(filter(str.isalpha, seq)).upper() for seq in sequences]


def write_to_fasta(sequences, filename, description_prefix="seq"):
    with open(filename, "w") as file:
        for index, sequence in enumerate(sequences, start=1):
            file.write(f">{description_prefix}_{index}\n")
            file.write(f"{sequence}\n")


def loading_esm():

    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    return batch_converter, esm_model, alphabet

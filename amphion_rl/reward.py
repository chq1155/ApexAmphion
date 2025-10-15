import os
import gzip
import shutil
import subprocess
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import modlamp.analysis as manalysis


def process_batch(batch: List[str]) -> pd.DataFrame:
    with open("pred.fa", "w") as f, gzip.open("pred.faa.gz", "wb") as f_out:
        for i, seq in enumerate(batch):
            f.write(f">seq_{i}\n{seq}\n")
        f.seek(0)
        shutil.copyfileobj(f, f_out)

    run_macrel("pred.faa.gz", "out_folder")

    with gzip.open("out_folder/macrel.out.prediction.gz", "rt") as f:
        next(f)
        df = pd.read_csv(
            f, sep="\t", usecols=["AMP_probability", "Hemolytic_probability"]
        )

    for file in ["pred.fa", "pred.faa.gz"]:
        os.remove(file)
    shutil.rmtree("out_folder")

    return df


def generate_mask(sequences: List[str], mode: str = "seq") -> torch.Tensor:
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_length = lengths.max() + (2 if mode == "bos" else 0)
    return torch.arange(max_length).expand(
        len(sequences), max_length
    ) < lengths.unsqueeze(1)


def inter(esm_embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_embedding = esm_embedding * mask.unsqueeze(-1)
    return masked_embedding.sum(dim=1, keepdim=True) / mask.sum(
        dim=1, keepdim=True
    ).unsqueeze(-1)


def encode(
    seq: List[str], esm_model, batch_converter, alphabet, device: str = "cuda"
) -> torch.Tensor:
    esm_masks = generate_mask(seq).to(device)
    _, _, batch_tokens = batch_converter(
        [("protein_{i}", s) for i, s in enumerate(seq)]
    )
    with torch.no_grad():
        results = esm_model(
            batch_tokens.to(device), repr_layers=[6], return_contacts=False
        )
    return inter(results["representations"][6][:, 1:-1, :], esm_masks)


def edistance(
    batch_embedding: torch.Tensor, target_embedding: torch.Tensor
) -> torch.Tensor:
    return torch.norm(batch_embedding - target_embedding, dim=-1)


def reward_amp(
    target_embedding: torch.Tensor,
    seq: List[str],
    esm_model,
    batch_converter,
    alphabet,
    device: str = "cuda",
) -> torch.Tensor:
    return edistance(
        encode(seq, esm_model, batch_converter, alphabet, device), target_embedding
    )


def reward_amp_cls(
    seq: List[str], esm_model, batch_converter, alphabet, cls1, device: str = "cuda"
) -> torch.Tensor:
    """
    Score sequences with the AMP classification head.

    Returns a tensor with positive values favouring likely AMPs and negative
    values penalising unlikely ones.
    """
    with torch.no_grad():
        esm_emb = encode(seq, esm_model, batch_converter, alphabet, device)
        score1 = cls1(esm_emb)
        mask1 = score1 >= 0.5
        output1 = score1 - 0.5
        print(f"Total number of qualified samples: {mask1.sum()}/{len(seq)}")
    return output1.view(-1).detach()


def calculate_physchem_prop(sequences: List[str]) -> Dict[str, List[float]]:
    global_analysis = manalysis.GlobalAnalysis(sequences)
    global_analysis.calc_H(scale="eisenberg")
    global_analysis.calc_charge()

    descriptor = manalysis.GlobalDescriptor(sequences)
    descriptor.isoelectric_point()

    moment = manalysis.PeptideDescriptor(sequences, "eisenberg")
    moment.calculate_moment()

    return {
        "length": [len(seq) for seq in sequences],
        "hydrophobicity": global_analysis.H[0].tolist(),
        "hydrophobic_moment": moment.descriptor.flatten().tolist(),
        "charge": global_analysis.charge[0].tolist(),
        "isoelectric_point": descriptor.descriptor.flatten().tolist(),
    }


def prop_reward(seqs: List[str]) -> torch.Tensor:
    # AMPGen-property reward function
    prop_dict = calculate_physchem_prop(seqs)

    weights = torch.tensor([[15, 3, 1.5, 0.1, -6]], dtype=torch.float32)
    metric_keys = [
        "length",
        "hydrophobicity",
        "hydrophobic_moment",
        "charge",
        "isoelectric_point",
    ]
    properties = torch.stack(
        [torch.tensor(prop_dict[key], dtype=torch.float32) for key in metric_keys]
    )
    retval = torch.matmul(weights, properties).view(-1)
    return retval


class CompositeReward:
    """
    Combine classification and physicochemical rewards at a configurable ratio.

    The combined reward can be used directly with PPO trainers expecting a tensor
    of shape (batch_size,). The individual components are cached on every call
    and are accessible via ``last_components`` for logging or debugging.
    """

    def __init__(
        self,
        esm_model,
        batch_converter,
        alphabet,
        cls_model,
        cls_weight: float = 0.5,
        prop_weight: float = 0.5,
        normalize_components: bool = True,
        device: str = "cuda",
    ):
        total = cls_weight + prop_weight
        if total <= 0:
            raise ValueError("cls_weight and prop_weight must sum to a positive value.")

        self.cls_weight = cls_weight / total
        self.prop_weight = prop_weight / total
        self.normalize_components = normalize_components
        self.device = device

        self.esm_model = esm_model.eval()
        self.batch_converter = batch_converter
        self.alphabet = alphabet
        self.cls_model = cls_model.eval()

        self.last_components: Dict[str, torch.Tensor] = {}
        self._eps = 1e-8

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.normalize_components:
            return scores
        mean = scores.mean()
        std = scores.std(unbiased=False)
        if std < self._eps:
            return scores - mean
        return (scores - mean) / (std + self._eps)

    def __call__(self, seqs: List[str]) -> torch.Tensor:
        cls_scores = reward_amp_cls(
            seqs,
            esm_model=self.esm_model,
            batch_converter=self.batch_converter,
            alphabet=self.alphabet,
            cls1=self.cls_model,
            device=self.device,
        ).detach()

        prop_scores = prop_reward(seqs).to(cls_scores.device, dtype=cls_scores.dtype)

        cls_norm = self._normalize(cls_scores)
        prop_norm = self._normalize(prop_scores)

        combined = self.cls_weight * cls_norm + self.prop_weight * prop_norm

        self.last_components = {
            "cls_raw": cls_scores.detach().cpu(),
            "prop_raw": prop_scores.detach().cpu(),
            "cls_norm": cls_norm.detach().cpu(),
            "prop_norm": prop_norm.detach().cpu(),
            "combined": combined.detach().cpu(),
        }

        return combined

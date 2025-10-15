import argparse
import pickle

import esm
import pandas as pd
import torch
from tqdm import tqdm


def load_model(model_name="esm2_t6_8M_UR50D"):
    model, alphabet = getattr(esm.pretrained, model_name)()
    return model.eval().cuda(), alphabet.get_batch_converter()


def get_embedding(model, batch_converter, sequence):
    batch_tokens = batch_converter([("protein", sequence)])[2].cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
    return results["representations"][6][0, 1 : len(sequence) + 1].mean(0)


def process_dataframe(df, model, batch_converter):
    result_dict = {
        "seq": df["Sequence"].tolist(),
        "esm_emb": [],
        "label": df["Label"].tolist(),
    }

    for sequence in tqdm(df["Sequence"], desc="Processing sequences"):
        embedding = get_embedding(model, batch_converter, sequence)
        result_dict["esm_emb"].append(embedding.cpu())

    return result_dict


def process_csv(csv_path, output_path, model, batch_converter):
    df = pd.read_csv(csv_path)
    result_dict = process_dataframe(df, model, batch_converter)
    with open(output_path, "wb") as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-i", "--input")
    args = parser.parse_args()

    model, batch_converter = load_model()
    csv_path = args.input
    output_path = csv_path + ".pkl"
    process_csv(csv_path, output_path, model, batch_converter)

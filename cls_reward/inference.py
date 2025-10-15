import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from mlp import MLP


class SequenceDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data["seq"])

    def __getitem__(self, index):
        return self.data["esm_emb"][index].clone().detach()


def get_dataloader(pkl_file, batch_size=32, shuffle=False, num_workers=2):
    return DataLoader(
        SequenceDataset(pkl_file),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def test(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(emb.to(device)).cpu().numpy()
                for emb in tqdm(dataloader, desc="Testing", unit="batch")
            ]
        )


def run_inference(
    test_pkls,
    best_model_path,
    batch_size=32,
    input_dim=320,
    hidden_dim=128,
    output_dim=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(best_model_path))
    return [test(model, get_dataloader(pkl, batch_size), device) for pkl in test_pkls]


def plot_probability_distributions(
    predictions, custom_titles, kde_colors, fill_colors, mean_line_color, main_title
):
    n = len(predictions)
    rows, cols = int(np.ceil(n / 2)), 2
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(main_title, fontsize=20)
    axs = axs.reshape(rows, -1)

    for pred, title, kde_color, fill_color, ax in zip(
        predictions, custom_titles, kde_colors, fill_colors, axs.flatten()
    ):
        valid_pred = pred[~np.isnan(pred) & ~np.isinf(pred)].flatten()
        mean_value = valid_pred.mean()

        if len(valid_pred) > 1:
            gmm = GaussianMixture(
                n_components=min(3, len(np.unique(valid_pred))), random_state=42
            ).fit(valid_pred.reshape(-1, 1))
            x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
            pdf = np.exp(gmm.score_samples(x_range))

            ax.plot(x_range, pdf, color=kde_color)
            ax.fill_between(x_range.flatten(), pdf, alpha=1.0, color=fill_color)
            ax.axvline(x=mean_value, color=mean_line_color, linestyle="--", linewidth=3)
            ax.text(
                0.02,
                0.95,
                f"Mean: {mean_value:.2f}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=18,
            )
        else:
            ax.set_title(f"{title}\n(Not enough data)", fontsize=18)

        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 16)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    for ax in axs.flatten()[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.2, hspace=0.3
    )
    plt.savefig("prediction_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(
        "Probability distribution functions have been plotted and saved as prediction_distributions.png"
    )


if __name__ == "__main__":
    test_pkls = [
        "sample_pkl_output/processed_train.pkl",
        "sample_pkl_output/processed_pepcvae.pkl",
        "sample_pkl_output/processed_hydramp.pkl",
        "sample_pkl_output/processed_ampgen1.pkl",
        "sample_pkl_output/processed_ampgen1_filter.pkl",
        "sample_pkl_output/processed_ampgen2_160_1w_nofilt.pkl",
        "sample_pkl_output/processed_ampgen2_filter.pkl",
        "sample_pkl_output/processed_ampgen_prop.pkl",
    ]
    best_model_path = "best_new_4.pth"

    predictions = run_inference(test_pkls, best_model_path)
    custom_titles = [
        "Training Data",
        "PepCVAE",
        "HydrAMP",
        "AMPGen",
        "AMPGen_filter",
        "AMPGen2",
        "AMPGen2_filter",
        "AMPGen_property",
    ]
    kde_colors = ["#576fa0"] * 8
    fill_colors = ["#a7b9d7"] * 8
    mean_line_color = "#b57979"
    main_title = ""

    plot_probability_distributions(
        predictions, custom_titles, kde_colors, fill_colors, mean_line_color, main_title
    )

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from .mlp import MLP, FocalLoss
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SequenceDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data["seq"])

    def __getitem__(self, index):
        return (
            self.data["esm_emb"][index].clone().detach(),
            self.data["label"][index],
        )


def get_dataloader(pkl_file, batch_size=32, shuffle=True, num_workers=2):
    return DataLoader(
        SequenceDataset(pkl_file),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for emb, label in tqdm(dataloader, desc="Training"):
        emb, label = emb.to(device), label.to(device)
        loss = criterion(model(emb), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for emb, label in tqdm(dataloader, desc="Validating"):
            emb, label = emb.to(device), label.to(device)
            output = model(emb)
            total_loss += criterion(output, label).item()
            all_preds.extend((output >= 0.4).float().cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
    }
    return metrics


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path,
):
    best_f1 = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_metrics = validate(model, val_dataloader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, "
            f"Val Loss: {val_metrics['loss']:.8f}, Val Accuracy: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, Val Recall: {val_metrics['recall']:.4f}, "
            f"Val Precision: {val_metrics['precision']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), save_path)
            print(
                f"Best model saved at epoch {epoch+1} with Val F1 Score: {best_f1:.4f}"
            )

    print(f"Training completed. Best model saved at: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader = get_dataloader(
        "/home/ubuntu/data/v2_data/train.csv.pkl", batch_size=32
    )
    val_dataloader = get_dataloader(
        "/home/ubuntu/data/v2_data/valid.csv.pkl", batch_size=32
    )

    model = MLP(input_dim=320, hidden_dim=128, output_dim=1).to(device)
    criterion = FocalLoss(alpha=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        device,
        num_epochs=25,
        save_path="/home/ubuntu/AMPGen_Product/models/cls_reward.pth",
    )


if __name__ == "__main__":
    main()

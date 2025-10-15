import torch
from torch.utils.data import DataLoader, Dataset


class AMPQueryDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tensor = torch.tensor([[1]], device="cuda:0")
        return {"input_ids": tensor}


def loading_dataset(steps, batch_size):

    dataset = AMPQueryDataset(steps * batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

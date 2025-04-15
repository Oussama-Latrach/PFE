import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.features = torch.tensor(data[:, :5], dtype=torch.float32)  # [N, 5]
        self.labels = torch.tensor(data[:, 5], dtype=torch.long) - 1  # Convert to 0-5

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # [5], [1]


def get_loaders(batch_size=32):
    train_dataset = PointCloudDataset("data/extracted_data/train_extracted.npy")
    val_dataset = PointCloudDataset("data/extracted_data/val_extracted.npy")
    test_dataset = PointCloudDataset("data/extracted_data/test_extracted.npy")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True if torch.cuda.is_available() else False)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True if torch.cuda.is_available() else False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True if torch.cuda.is_available() else False)

    return train_loader, val_loader, test_loader
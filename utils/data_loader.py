"""
Data loading utilities for point cloud classification.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    """Custom Dataset for point cloud classification."""

    def __init__(self, file_path):
        """
        Args:
            file_path: Path to .npy file containing point cloud data
        """
        data = np.load(file_path)
        # Features: x,y,z, return_number, number_of_returns
        self.features = torch.tensor(data[:, :5], dtype=torch.float32)
        # Labels: classification (converted to 0-based index)
        self.labels = torch.tensor(data[:, 5], dtype=torch.long) - 1

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_loaders(batch_size=32):
    """Create data loaders for train/val/test sets."""
    print(" Loading datasets...")

    # Initialize datasets
    train_dataset = PointCloudDataset("E:/cours_geomatique_3eme_annee/PFE/pratique/projet2/data/extracted_data/train_extracted.npy")
    val_dataset = PointCloudDataset("E:/cours_geomatique_3eme_annee/PFE/pratique/projet2/data/extracted_data/val_extracted.npy")
    test_dataset = PointCloudDataset("E:/cours_geomatique_3eme_annee/PFE/pratique/projet2/data/extracted_data/test_extracted.npy")

    print(f" Dataset sizes:")
    print(f"  - Train: {len(train_dataset):,} points")
    print(f"  - Val:   {len(val_dataset):,} points")
    print(f"  - Test:  {len(test_dataset):,} points")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available())

    print(f"\n Batch size: {batch_size}")
    print(f" Train batches: {len(train_loader)}")
    print(f" Val batches:   {len(val_loader)}")
    print(f" Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader
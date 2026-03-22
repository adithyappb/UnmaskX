import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image


def load_face_data(batch_size: int, data_dir: str) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = FaceDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

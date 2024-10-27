import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def load_data(batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = FaceDataset(data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

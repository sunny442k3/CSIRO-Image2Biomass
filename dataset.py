import numpy as np
import cv2 
from torch.utils.data import Dataset
import torch
import pathlib 

ALL_TARGET_COLS = ['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
SCALES = torch.tensor([157.9836, 83.8407, 71.7865, 157.9836, 185.7000]).float()

class BiomassDataset(Dataset):
    def __init__(self, df, transform):
        self.root = pathlib.Path("./datasets")
        self.df = df
        self.transform = transform
        self.paths = df['id'].values
        self.labels = torch.from_numpy(df[ALL_TARGET_COLS].values).float()
        self.labels /= SCALES.unsqueeze(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.root / self.paths[idx]
        img = cv2.imread(path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new0 = self.transform(image=img[:, :1000])['image']
        new1 = self.transform(image=img[:, 1000:])['image']
        label = self.labels[idx] 
        return new0, new1, label
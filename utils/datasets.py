import os
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self,csv_file, img_dir, label_dir, transform= None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        annotation_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        image = Image.open(img_path)
        boxes = self._parse_voc_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        return  image, boxes

    def _parse_voc_annotation(self, annotation_path: str) -> torch.tensor:
        data = pd.read_csv(annotation_path, sep=" ", header=None)
        boxes = []
        for _, row in data.iterrows():         
            boxes.append(row)
        
        return torch.tensor(boxes)
    
    def collate_fn(self, minibatch):
        images, targets = zip(*minibatch)
        images = torch.stack(images, dim=0)
        return images, targets





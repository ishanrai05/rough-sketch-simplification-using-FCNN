from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, input_path, target_path, height=424, width=424, transform=None):
        self.transform = transform
        self.input_img = input_path
        self.target_img = target_path
        self.height = height
        self.width = width

    def __len__(self):
        return len(self.input_img)
    
    def preprocess(self, img_path):
        img = Image.open(img_path)
        img = img.convert('L')
        img = img.resize((self.height, self.width))
        img = np.asarray(img)
        return img

    def __getitem__(self, index):
        
        X = self.preprocess(self.input_img[index])
        y = self.preprocess(self.target_img[index])

        if self.transform:
            X = self.transform(X)
            y = self.transform(y)

        return X, y
import torch
from torch.utils.data import Dataset
import numpy as np


class BeitLocalDataset(Dataset):
    def __init__(self, size):
        super(BeitLocalDataset, self).__init__()
        self.random_sample = self._get_sample()
        self._size = size

    def __getitem__(self, item):
        return self.random_sample

    def __len__(self):
        return self._size

    def _get_sample(self):
        img = torch.rand(size=[1, 224, 224], dtype=torch.float32)
        image_for_tokenizer = torch.rand(size=[1, 224, 224], dtype=torch.float32)
        return {'img': img,
                'image_for_tokenizer': image_for_tokenizer}

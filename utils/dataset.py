import os, csv, PIL

import numpy as np
from torch.utils.data import Dataset


class iMatDataset(Dataset):
    def __init__(self, data_type, transform=None):
        if data_type == 'train':
            self.img_dir = './data/train/'
            self.label_dir = './data/train.csv'
        elif data_type == 'val':
            self.img_dir = './data/val/'
            self.label_dir = './data/val.csv'
        elif data_type == 'test':
            self.img_dir = './data/test/'
            self.label_dir = './data/test.csv'

        ids = []
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg'):
                id_, _ = img_file.split('.')
                ids.append(int(id_))
        self.ids = sorted(ids)

        self.id2label = {}
        with open(self.label_dir, 'r') as csvfile:
            for line in csv.reader(csvfile):
                id_, *label = list(map(int, line))
                self.id2label[id_] = label

        self.n_labels = 228
        self.transform = transform

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        img_path = os.path.join(self.img_dir, str(id_)+'.jpg')
        img = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        labels_idx = [label-1 for label in self.id2label[id_]]
        labels = np.zeros(self.n_labels)
        labels[labels_idx] = 1.0

        return img, labels

    def __len__(self):
        return len(self.ids)

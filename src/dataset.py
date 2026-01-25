import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LungSegDataset(Dataset):
    def __init__(self, img_dir, mask_root):
        self.img_dir = img_dir
        self.mask_root = mask_root
        self.files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        img_path = os.path.join(self.img_dir, f)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        
        l_path = os.path.join(self.mask_root, 'leftMask', f)
        r_path = os.path.join(self.mask_root, 'rightMask', f)
        mask = np.zeros((224, 224), dtype=np.float32)
        
        if os.path.exists(l_path):
            mask = np.maximum(mask, cv2.resize(cv2.imread(l_path, 0), (224, 224)))
        if os.path.exists(r_path):
            mask = np.maximum(mask, cv2.resize(cv2.imread(r_path, 0), (224, 224)))
            
        mask = (mask > 0).astype(np.float32)
        img = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask

class NIHDataset(Dataset):
    def __init__(self, csv, root):
        self.df = pd.read_csv(csv)
        self.path_map = {}
        for r, d, f in os.walk(root):
            for file in f: 
                if file.endswith('.png'): self.path_map[file] = os.path.join(r, file)
        self.df = self.df[self.df['Image Index'].isin(self.path_map.keys())]
        self.labels = ['Infiltration', 'Consolidation', 'Pneumonia', 'Mass', 'Nodule']
        
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.path_map[row['Image Index']]).convert('RGB')
        img = img.resize((224,224))
        lbl = torch.zeros(5)
        for i, l in enumerate(self.labels):
            if l in row['Finding Labels']: lbl[i] = 1.0
        return transforms.ToTensor()(img), lbl

class ShenzhenDataset(Dataset):
    def __init__(self, img_root, txt_root=None):
        self.img_root = img_root
        self.txt_root = txt_root
        self.files = [f for f in os.listdir(img_root) if f.endswith('.png')]
        self.tfm = transforms.Compose([
            transforms.Resize((224,224)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
                                    
    def __len__(self): return len(self.files)
    
    def __getitem__(self, i):
        filename = self.files[i]
        path = os.path.join(self.img_root, filename)
        label = 1 if '_1.png' in filename else 0 
        text = "Clinical details unavailable."
        if self.txt_root:
            txt_filename = filename.replace('.png', '.txt')
            txt_path = os.path.join(self.txt_root, txt_filename)
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().strip()
                except: pass
        return self.tfm(Image.open(path).convert('RGB')), label, text, filename

def collate_fn(batch):
    imgs, lbls, texts, fnames = zip(*batch)
    return torch.stack(imgs), torch.tensor(lbls), texts, fnames

import os

# Define the folder structure
folders = [
    "data",
    "saved_models",
    "src"
]

# Define the file contents
files = {}

# 1. ROOT FILES
files["requirements.txt"] = """torch
torchvision
numpy
pandas
opencv-python-headless
matplotlib
pillow
scikit-learn
timm
segmentation-models-pytorch
grad-cam"""

files["README.md"] = """# Chest X-Ray Analysis using U-Net + ViT + RAG

## Project Structure
* `src/config.py`: Configuration and Paths
* `src/dataset.py`: PyTorch Dataset classes
* `src/models.py`: Model architecture definitions
* `src/train.py`: Training loops for U-Net and ViT
* `src/utils.py`: Helper functions (visualization, RAG)
* `main.py`: Entry point for the project

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py`
"""

files["main.py"] = """from src.config import DEVICE, locate_datasets
from src.train import train_segmentation, pretrain_nih, finetune_shenzhen
from src.utils import visualize_explainability, build_rag_db

if __name__ == "__main__":
    print(f"🚀 MEDICAL AI PIPELINE STARTED ON {DEVICE}")
    
    # 1. Locate Data (Change path if not on Kaggle)
    paths = locate_datasets(start_path='/kaggle/input') 
    
    # 2. Train Segmentation (U-Net)
    unet_model = train_segmentation(paths)
    
    # 3. Pre-train ViT on NIH
    pretrain_nih(paths)
    
    # 4. Fine-tune ViT on Shenzhen
    vit_model, val_dl = finetune_shenzhen(paths)
    
    if vit_model and val_dl:
        # 5. Show Visuals (Grad-CAM + Segmentation + Text)
        visualize_explainability(vit_model, unet_model, val_dl)
        
        # 6. Build RAG Database
        build_rag_db(vit_model, val_dl)
"""

# 2. SOURCE CODE FILES (src/)

files["src/__init__.py"] = ""

files["src/config.py"] = """import torch
import os
import warnings

warnings.filterwarnings('ignore')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def locate_datasets(start_path='/kaggle/input'):
    print("\\n🔍 LOCATING DATASETS...")
    paths = {'nih': {}, 'shenzhen': {}, 'montgomery': {}}
    
    for root, dirs, files in os.walk(start_path):
        if 'Data_Entry_2017.csv' in files:
            paths['nih']['csv'] = os.path.join(root, 'Data_Entry_2017.csv')
            paths['nih']['root'] = root
        
        if 'ChinaSet_AllFiles' in root and 'CXR_png' in dirs:
            paths['shenzhen']['img'] = os.path.join(root, 'CXR_png')
            if 'ClinicalReadings' in dirs:
                paths['shenzhen']['txt'] = os.path.join(root, 'ClinicalReadings')
        
        if 'MontgomerySet' in root and 'CXR_png' in dirs:
            paths['montgomery']['img'] = os.path.join(root, 'CXR_png')
        if 'ManualMask' in root and 'leftMask' in dirs:
            paths['montgomery']['mask'] = root

    print(f"   ✅ NIH Found: {bool(paths['nih'])}")
    print(f"   ✅ Shenzhen Found: {bool(paths['shenzhen'])}")
    print(f"   ✅ Montgomery Found: {bool(paths['montgomery'])}")
    return paths
"""

files["src/dataset.py"] = """import os
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
"""

files["src/models.py"] = """import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
from .config import DEVICE

def build_unet():
    model = smp.Unet('resnet18', classes=1, activation=None).to(DEVICE)
    return model

def build_vit(pretrained=True, num_classes=5):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)
"""

files["src/train.py"] = """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from .config import DEVICE
from .dataset import LungSegDataset, NIHDataset, ShenzhenDataset, collate_fn
from .models import build_unet, build_vit
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def train_segmentation(paths):
    if not paths['montgomery'].get('mask'):
        print("⚠️ Skipping Segmentation (Montgomery masks missing)")
        return None
        
    print("\\n🏗️ Training U-Net for Lung Segmentation...")
    unet = build_unet()
    ds = LungSegDataset(paths['montgomery']['img'], paths['montgomery']['mask'])
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    
    opt = optim.Adam(unet.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    
    unet.train()
    for epoch in range(3): 
        epoch_loss = 0
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = crit(unet(imgs), masks)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"   Epoch {epoch+1} Loss: {epoch_loss/len(dl):.4f}")
        
    torch.save(unet.state_dict(), 'saved_models/unet_lung.pth')
    print("✅ U-Net Saved.")
    return unet

def pretrain_nih(paths):
    if not paths['nih'].get('csv'):
        print("⚠️ Skipping NIH Pre-training (Dataset missing)")
        return None
        
    print("\\n🏗️ Pre-training ViT on NIH (14-Class)...")
    vit = build_vit(pretrained=True, num_classes=5)
    
    nih_ds = NIHDataset(paths['nih']['csv'], paths['nih']['root'])
    sub_ds, _ = random_split(nih_ds, [min(2000, len(nih_ds)), len(nih_ds)-min(2000, len(nih_ds))])
    dl = DataLoader(sub_ds, batch_size=32, shuffle=True)
    
    opt = optim.AdamW(vit.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    
    vit.train()
    for i, (imgs, lbls) in enumerate(dl):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        loss = crit(vit(imgs), lbls)
        loss.backward()
        opt.step()
        if i % 20 == 0: print(f"   Batch {i} Loss: {loss.item():.4f}")
        
    torch.save(vit.state_dict(), 'saved_models/vit_nih.pth')
    print("✅ NIH Weights Saved.")
    return vit

def finetune_shenzhen(paths):
    if not paths['shenzhen'].get('img'):
        print("❌ Shenzhen dataset missing."); return None, None

    print("\\n🏗️ Fine-tuning ViT on Shenzhen...")
    ds = ShenzhenDataset(paths['shenzhen']['img'], paths['shenzhen'].get('txt'))
    
    train_size = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Load Model (Load NIH weights if available)
    model = build_vit(pretrained=False, num_classes=5)
    if os.path.exists('saved_models/vit_nih.pth'):
        model.load_state_dict(torch.load('saved_models/vit_nih.pth'))
        print("   Using NIH Pre-trained Weights.")
    else:
        print("   Using ImageNet Weights.")
        model = build_vit(pretrained=True, num_classes=5)
        
    model.head = nn.Linear(768, 2) # Binary
    model = model.to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=5e-5)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    print("   Training (Demo)...")
    for i, (imgs, lbls, _, _) in enumerate(train_dl):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(imgs), lbls)
        loss.backward()
        opt.step()
        
    torch.save(model.state_dict(), 'saved_models/vit_final.pth')
    print("🎉 Final ViT Model Saved.")
    
    # Metrics
    print("\\n📊 CALCULATING BENCHMARK METRICS...")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbls, _, _ in val_dl:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    print(f"🎯 Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}")
    
    return model, val_dl
"""

files["src/utils.py"] = """import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from .config import DEVICE

def reshape_transform(tensor):
    result = tensor[:, 1:, :].reshape(tensor.size(0), 14, 14, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def visualize_explainability(vit_model, unet_model, val_dl):
    print("\\n🔬 GENERATING MULTI-MODAL VISUALIZATION...")
    target_layers = [vit_model.norm]
    cam = GradCAM(model=vit_model, target_layers=target_layers, reshape_transform=reshape_transform)
    found = False
    for imgs, lbls, texts, fnames in val_dl:
        for i in range(len(lbls)):
            if lbls[i] == 1: # TB Positive
                input_tensor = imgs[i].unsqueeze(0).to(DEVICE)
                fname = fnames[i]
                text = texts[i]
                
                # ViT
                vit_model.eval()
                output = vit_model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0, :]
                
                # U-Net
                lung_mask_viz = np.zeros((224, 224))
                if unet_model:
                    unet_model.eval()
                    with torch.no_grad():
                        mask_logits = unet_model(input_tensor)
                        lung_mask_viz = (torch.sigmoid(mask_logits).cpu().numpy()[0, 0] > 0.5).astype(np.float32)
                
                # Plot
                rgb_img = np.clip(imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, 0, 1)
                heatmap_viz = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                seg_viz = rgb_img.copy()
                contours, _ = cv2.findContours((lung_mask_viz * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(seg_viz, contours, -1, (0, 1, 0), 2)
                
                fig = plt.figure(figsize=(16, 10))
                ax1 = plt.subplot(2, 2, 1); ax1.imshow(rgb_img); ax1.set_title(f"Original: {fname}"); ax1.axis('off')
                ax2 = plt.subplot(2, 2, 2); ax2.imshow(seg_viz); ax2.set_title("U-Net Segmentation"); ax2.axis('off')
                ax3 = plt.subplot(2, 2, 3); ax3.imshow(heatmap_viz); ax3.set_title(f"ViT Attention (Prob: {prob*100:.1f}%)"); ax3.axis('off')
                ax4 = plt.subplot(2, 2, 4); ax4.axis('off'); ax4.text(0, 0.5, f"REPORT:\\n{text}", fontsize=11, wrap=True)
                plt.tight_layout(); plt.show()
                print(f"✅ Visualization generated for {fname}")
                found = True
                break
        if found: break

def build_rag_db(model, val_dl):
    print("\\n💾 BUILDING RAG VECTOR DATABASE...")
    db_vectors, db_texts, db_filenames = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls, texts, fnames in val_dl:
            imgs = imgs.to(DEVICE)
            features = model.forward_features(imgs)[:, 0, :]
            db_vectors.append(features.cpu().numpy())
            db_texts.extend(texts)
            db_filenames.extend(fnames)
            
    final_vectors = np.concatenate(db_vectors, axis=0)
    np.save('saved_models/rag_vectors.npy', final_vectors)
    np.save('saved_models/rag_texts.npy', np.array(db_texts))
    np.save('saved_models/rag_filenames.npy', np.array(db_filenames))
    print(f"✅ Database Built: {final_vectors.shape[0]} entries.")
"""

# Execution of folder creation
print("⚙️ Setting up Professional Project Structure...")
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"   Created folder: {folder}/")

for filename, content in files.items():
    with open(filename, 'w') as f:
        f.write(content)
    print(f"   Created file:   {filename}")

print("\n✅ Project Setup Complete! You can now run: python main.py")
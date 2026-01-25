import torch
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
        
    print("\n🏗️ Training U-Net for Lung Segmentation...")
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
        
    print("\n🏗️ Pre-training ViT on NIH (14-Class)...")
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

    print("\n🏗️ Fine-tuning ViT on Shenzhen...")
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
    print("\n📊 CALCULATING BENCHMARK METRICS...")
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

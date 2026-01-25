import torch
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
    print("\n🔬 GENERATING MULTI-MODAL VISUALIZATION...")
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
                ax4 = plt.subplot(2, 2, 4); ax4.axis('off'); ax4.text(0, 0.5, f"REPORT:\n{text}", fontsize=11, wrap=True)
                plt.tight_layout(); plt.show()
                print(f"✅ Visualization generated for {fname}")
                found = True
                break
        if found: break

def build_rag_db(model, val_dl):
    print("\n💾 BUILDING RAG VECTOR DATABASE...")
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

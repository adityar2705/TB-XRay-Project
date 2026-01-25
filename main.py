from src.config import DEVICE, locate_datasets
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

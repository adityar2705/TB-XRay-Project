import torch
import os
import warnings

warnings.filterwarnings('ignore')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def locate_datasets(start_path='/kaggle/input'):
    print("\n🔍 LOCATING DATASETS...")
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

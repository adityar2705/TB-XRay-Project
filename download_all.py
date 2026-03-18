import os
import shutil
import urllib.request
import zipfile
import kagglehub

print("🚀 INITIATING FULL AUTOMATED DATA DOWNLOAD...")
base_dir = "./input"
os.makedirs(base_dir, exist_ok=True)

# --- 1. NIH DATASET (via KaggleHub) ---
print("\n📦 1/3: Downloading NIH Dataset via Kaggle...")
nih_cache = kagglehub.dataset_download("nih-chest-xrays/data")
nih_target = os.path.join(base_dir, "NIH_ChestXray")
os.makedirs(nih_target, exist_ok=True)
print("🚚 Moving NIH files to input folder...")
shutil.copytree(nih_cache, nih_target, dirs_exist_ok=True)
print("✅ NIH Dataset ready.")

# --- 2. SHENZHEN DATASET (via Direct NLM Link) ---
sz_url = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/ShenzhenHospitalCXRSet.zip"
sz_zip = "shenzhen.zip"
print("\n📦 2/3: Downloading Shenzhen Dataset...")
urllib.request.urlretrieve(sz_url, sz_zip)
print("📂 Extracting Shenzhen files...")
with zipfile.ZipFile(sz_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)
os.remove(sz_zip) # Clean up the zip file
print("✅ Shenzhen Dataset ready.")

# --- 3. MONTGOMERY DATASET (via Direct NLM Link) ---
mc_url = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet.zip"
mc_zip = "montgomery.zip"
print("\n📦 3/3: Downloading Montgomery Dataset...")
urllib.request.urlretrieve(mc_url, mc_zip)
print("📂 Extracting Montgomery files...")
with zipfile.ZipFile(mc_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)
os.remove(mc_zip) # Clean up the zip file
print("✅ Montgomery Dataset ready.")

print("\n🎉 ALL DATA DOWNLOADED AND PERFECTLY STRUCTURED. YOU ARE READY TO RUN MAIN.PY!")
import os
import sys
import glob
import json
import subprocess
from pathlib import Path

def ensure_simpleitk():
    try:
        import SimpleITK as sitk
        print("✅ SimpleITK is already installed.")
    except ImportError:
        print("⚠️ SimpleITK not found. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SimpleITK"])
        print("✅ SimpleITK installed successfully.")

def fix_notebook():
    notebook_path = r'C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix DATA_DIR and remove the bad v1 path
        content = content.replace("DATA_DIR    = BASE_DIR / 'Dataset'", "DATA_DIR    = Path(r'C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset')")

        # Fix the CSV dictionary to use relative paths so we don't hit the unicode \U error
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\ADNIMERGE_09Mar2026.csv", "ADNIMERGE_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\DXSUM_09Mar2026.csv", "DXSUM_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\FAQ_09Mar2026.csv", "FAQ_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\MEDHIST_09Mar2026.csv", "MEDHIST_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\MMSE_09Mar2026.csv", "MMSE_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\NPI_09Mar2026.csv", "NPI_09Mar2026.csv")
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\Dataset\\\\VITALS_09Mar2026.csv", "VITALS_09Mar2026.csv")
        
        # Exact string from the typo:
        content = content.replace("C:\\\\Users\\\\Rathish K\\\\Documents\\\\ML\\\\DatasetMRI_metadata_with_VISCODE.csv", "MRI_metadata_with_VISCODE.csv")

        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Notebook {notebook_path} successfully updated to fix unicode and path errors.")
    except Exception as e:
        print(f"⚠️ Could not update notebook: {e}")

def convert_dicoms():
    import SimpleITK as sitk
    DATASET_DIR = Path(r"C:\Users\Rathish K\Documents\ML\Dataset\ADNI_T1_Baseline_MRI\ADNI")
    print(f"\n🔍 Scanning for DICOM directories in {DATASET_DIR}...")
    
    count = 0
    for root, dirs, files in os.walk(DATASET_DIR):
        dcm_files = glob.glob(os.path.join(root, "*.dcm"))
        if len(dcm_files) > 0:
            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                if not dicom_names:
                    continue
                
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                
                out_path = os.path.join(root, "converted_mri.nii.gz")
                sitk.WriteImage(image, out_path)
                print(f"  -> Converted: {root} to {out_path}")
                count += 1
            except Exception as e:
                print(f"  -> Failed to convert {root}: {e}")

    print(f"\n✅ Conversion complete. Successfully converted {count} MRI scans to .nii.gz format.")

if __name__ == "__main__":
    ensure_simpleitk()
    fix_notebook()
    convert_dicoms()

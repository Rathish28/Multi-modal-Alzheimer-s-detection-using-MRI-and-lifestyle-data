import os
import glob
from pathlib import Path
import SimpleITK as sitk

# Paths
DATASET_DIR = Path(r"C:\Users\Rathish K\Documents\ML\Dataset\ADNI_T1_Baseline_MRI\ADNI")

print(f"Scanning for DICOM directories in {DATASET_DIR}...")
count = 0

# Walk through all directories in the ADNI folder
for root, dirs, files in os.walk(DATASET_DIR):
    # If the directory contains .dcm files, process it
    dcm_files = glob.glob(os.path.join(root, "*.dcm"))
    if len(dcm_files) > 0:
        print(f"Processing {len(dcm_files)} DICOM files in: {root}")
        
        try:
            # Read the DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(root)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Create a path for the output NIfTI file
            # E.g., save it as "converted_mri.nii.gz" in the same folder where the DICOMs are
            out_path = os.path.join(root, "converted_mri.nii.gz")
            
            # Save the NIfTI (.nii.gz) file
            sitk.WriteImage(image, out_path)
            print(f"  -> Saved NIfTI: {out_path}")
            count += 1
            
        except Exception as e:
            print(f"  -> Failed to convert {root}: {e}")

print(f"\n✅ Conversion complete. Successfully converted {count} MRI scans to .nii.gz format.")

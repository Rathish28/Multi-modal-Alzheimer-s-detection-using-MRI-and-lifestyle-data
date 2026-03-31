import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    if "def compute_gradcam_3d(" in source and "zoom(" not in source:
        # We need to resize the low-resolution heatmap back up to the original MRI shape
        patch = """
    import scipy.ndimage
    cam       = (weights[:, None, None, None] * feat).sum(0)  # (D, H, W)
    cam       = np.maximum(cam, 0)                   # ReLU
    cam_norm  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # NEW: Upscale the low-res heatmap back to the original MRI dimensions!
    target_shape = mri_tensor.shape[1:]  # (D, H, W)
    zoom_factors = [t / c for t, c in zip(target_shape, cam_norm.shape)]
    cam_resized  = scipy.ndimage.zoom(cam_norm, zoom_factors, order=1)
    
    return cam_resized
"""
        
        # Replace the end of the function with our patched return
        import re
        old_end = re.search(r"cam       = \(weights\[:, None, None, None\] \* feat\).sum\(0\).*?return cam_norm\n", source, re.DOTALL)
        if old_end:
            source = source.replace(old_end.group(0), patch.lstrip())
            
        nb['cells'][idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Grad-CAM spatial resolution upscaling patch injected!")

import json
import os

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        # Look for the buggy loop
        for j, line in enumerate(source):
            if "for ax in [2, 3, 4]:" in line:
                source[j] = line.replace("for ax in [2, 3, 4]:", "for ax in [1, 2, 3]:")
                found = True
                
        if found:
            nb['cells'][i]['source'] = source
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully: Fixed the dataset augmentation axes in Cell 11.")
else:
    print("Could not find the buggy line. Please check manually.")

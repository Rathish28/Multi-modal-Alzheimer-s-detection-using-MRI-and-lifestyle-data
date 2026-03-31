import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    # We need to replace all occurrences of `roc_auc_score` that don't already have `labels=`
    # with `roc_auc_score(..., labels=[0, 1, 2, 3])`
    
    # Pattern to search: "roc_auc_score(..., multi_class='ovr', average='macro')"
    if "roc_auc_score(" in source and "labels=" not in source:
        # Easy straight string replacements for the standard usages in the code
        source = source.replace(
            "multi_class='ovr', average='macro')",
            "multi_class='ovr', average='macro', labels=[0, 1, 2, 3])"
        )
        nb['cells'][idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("roc_auc_score calls patched successfully to accept 4 classes.")

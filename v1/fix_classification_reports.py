import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    if "classification_report(" in source and "labels=[0, 1, 2, 3]" not in source:
        source = source.replace(
            "classification_report(y_test_eval, ens_pred,",
            "classification_report(y_test_eval, ens_pred, labels=[0, 1, 2, 3],"
        )
    
    if "confusion_matrix(" in source and "labels=[0, 1, 2, 3]" not in source:
        source = source.replace(
            "confusion_matrix(y_test_eval, ens_pred)",
            "confusion_matrix(y_test_eval, ens_pred, labels=[0, 1, 2, 3])"
        )
    
    nb['cells'][idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Classification report and Confusion matrix missing class constraints patched!")

import json
import re

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    # 1. Remove the labels=[0,1,2,3] we added earlier so it stops generating NaNs
    source = source.replace(", labels=[0, 1, 2, 3]", "")
    
    # 2. Replace the raw probability matrix with a filtered one dynamically based on present classes
    # Cell 18
    source = source.replace(
        "auc_xgb = roc_auc_score(y_e_test, xgb_proba, multi_class='ovr', average='macro')",
        "valid_cls = np.unique(y_e_test)\nauc_xgb = roc_auc_score(y_e_test, xgb_proba[:, valid_cls] / xgb_proba[:, valid_cls].sum(1, keepdims=True), multi_class='ovr', average='macro')"
    )
    # Cell 19
    source = source.replace(
        "auc_xtab = roc_auc_score(y_test, xgb_tab_proba, multi_class='ovr', average='macro')",
        "valid_cls = np.unique(y_test)\nauc_xtab = roc_auc_score(y_test, xgb_tab_proba[:, valid_cls] / xgb_tab_proba[:, valid_cls].sum(1, keepdims=True), multi_class='ovr', average='macro')"
    )
    # Cell 20
    source = source.replace(
        "auc_snn = roc_auc_score(y_e_test, snn_proba, multi_class='ovr', average='macro')",
        "valid_cls = np.unique(y_e_test)\nauc_snn = roc_auc_score(y_e_test, snn_proba[:, valid_cls] / snn_proba[:, valid_cls].sum(1, keepdims=True), multi_class='ovr', average='macro')"
    )
    # Cell 21
    source = source.replace(
        "auc_fus = roc_auc_score(y_test_eval, fusion_proba, multi_class='ovr', average='macro')",
        "valid_cls = np.unique(y_test_eval)\nauc_fus = roc_auc_score(y_test_eval, fusion_proba[:, valid_cls] / fusion_proba[:, valid_cls].sum(1, keepdims=True), multi_class='ovr', average='macro')"
    )
    
    # Cell 22 -> compute_metrics
    if "def compute_metrics" in source:
        source = source.replace(
            "auc  = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')",
            "valid_cls = np.unique(y_true)\n        auc  = roc_auc_score(y_true, y_proba[:, valid_cls] / y_proba[:, valid_cls].sum(1, keepdims=True), multi_class='ovr', average='macro')"
        )
    
    nb['cells'][idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("AUC NaN math patched!")

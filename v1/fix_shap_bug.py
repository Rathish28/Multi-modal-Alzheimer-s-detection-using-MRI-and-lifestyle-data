import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    if "explainer    = shap.TreeExplainer(xgb_tab)" in source:
        workaround_code = """
import json
import xgboost as xgb
# WORKAROUND for known SHAP + XGBoost multiclass string bug:
xgb_tab.save_model('tmp_xgb.json')
with open('tmp_xgb.json', 'r') as _f:
    _js = json.load(_f)
# Force base_score to be a scalar string so older SHAP versions don't crash computing float("[0, 0, 0]")
_js["learner"]["learner_model_param"]["base_score"] = "0.5"
with open('tmp_xgb.json', 'w') as _f:
    json.dump(_js, _f)

# Load patched booster
patched_booster = xgb.Booster()
patched_booster.load_model('tmp_xgb.json')

explainer    = shap.TreeExplainer(patched_booster)
"""
        source = source.replace("explainer    = shap.TreeExplainer(xgb_tab)", workaround_code.strip())
        nb['cells'][idx]['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SHAP TreeExplainer base_score bug magically patched!")

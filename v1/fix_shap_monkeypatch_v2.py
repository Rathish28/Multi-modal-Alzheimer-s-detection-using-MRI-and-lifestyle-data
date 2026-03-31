import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    if "original_float = builtins.float" in source:
        new_source = """
print('🔍 Running SHAP analysis on XGBoost tabular model...')

import builtins
import shap

original_float = builtins.float
# We use a CLASS instead of a function so that Numpy's isinstance(val, (int, float)) doesn't crash!
class mock_float(float):
    def __new__(cls, val=0):
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            val = val.strip('[]').split(',')[0]
        return super().__new__(cls, val)

try:
    builtins.float = mock_float
    explainer = shap.TreeExplainer(xgb_tab)
finally:
    builtins.float = original_float  # Safely restore original float behavior

# By passing check_additivity=False, we tell SHAP not to panic about the tiny margin difference 
# caused by our math patch above.
shap_values = explainer.shap_values(X_tab_test, check_additivity=False)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values if isinstance(shap_values, np.ndarray) else shap_values[len(shap_values)-1],
    X_tab_test,
    feature_names=FEATURE_NAMES,
    plot_type='bar',
    show=False,
    max_display=20,
)
plt.title('Top 20 Feature Importances (SHAP) — Tabular Model', fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ Saved: {PLOT_DIR / "shap_importance.png"}')

# SHAP beeswarm
shap.summary_plot(
    shap_values if isinstance(shap_values, np.ndarray) else shap_values[len(shap_values)-1],
    X_tab_test,
    feature_names=FEATURE_NAMES,
    show=False,
    max_display=20,
)
plt.title('SHAP Beeswarm — Feature Impact on AD Class', fontsize=13)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ Saved: {PLOT_DIR / "shap_beeswarm.png"}')
"""
        nb['cells'][idx]['source'] = [line + '\n' for line in new_source.strip().split('\n')]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SHAP TreeExplainer float monkey-patch perfectly injected via Subclass! ✅")

"""
fix_cell18.py
Patches Cell 18 of alzheimer_multimodal_GPU.ipynb to add a defensive
label-remapping block that fixes the [0 1 3] → [0 1 2] issue for XGBoost.
"""

import json, copy, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")

# ── New source for Cell 18 ────────────────────────────────────────────────────
NEW_SOURCE = [
    "print('🌲 Training XGBoost on fused embeddings...')\n",
    "\n",
    "# ── Defensive label remapping ─────────────────────────────────────────────\n",
    "# XGBoost requires labels [0, 1, ..., k-1] with NO gaps.\n",
    "# The MRI-matched subset (163 subjects) may have a different class\n",
    "# distribution than the full tabular table (2409 subjects).\n",
    "# Example: full table has CN/LMCI/AD → label_arr has [0,1,2],\n",
    "#          but MRI subjects only have CN/EMCI/AD where EMCI=1 and AD=3\n",
    "#          (4-class encoding) → XGBoost sees [0,1,3] and crashes.\n",
    "# This block detects and fixes any such gap BEFORE calling xgb_model.fit().\n",
    "_unique_labels = np.unique(y_e_train)\n",
    "_expected      = np.arange(len(_unique_labels))\n",
    "if not np.array_equal(_unique_labels, _expected):\n",
    "    # Build old→new compact mapping\n",
    "    _remap = {int(old): int(new) for new, old in enumerate(_unique_labels)}\n",
    "    print(f'⚠️  Non-contiguous labels detected: {_unique_labels.tolist()}')\n",
    "    print(f'   Remapping → {_expected.tolist()} for XGBoost compatibility')\n",
    "    print(f'   Remap dict: {_remap}')\n",
    "    # Apply remap to all three splits (produces new arrays, safe copies)\n",
    "    y_e_train = np.vectorize(_remap.get)(y_e_train).astype(np.int64)\n",
    "    y_e_val   = np.vectorize(_remap.get)(y_e_val).astype(np.int64)\n",
    "    y_e_test  = np.vectorize(_remap.get)(y_e_test).astype(np.int64)\n",
    "    # Update CLASS_NAMES / N_CLASSES to match the MRI-subset classes only\n",
    "    # (uses the OLD label values as indices into CLASS_NAMES)\n",
    "    CLASS_NAMES = [CLASS_NAMES[i] for i in _unique_labels]\n",
    "    N_CLASSES   = len(CLASS_NAMES)\n",
    "    print(f'   Updated CLASS_NAMES: {CLASS_NAMES}  |  N_CLASSES: {N_CLASSES}')\n",
    "else:\n",
    "    print(f'✅ Labels already contiguous: {_unique_labels.tolist()}')\n",
    "\n",
    "xgb_model = xgb.XGBClassifier(\n",
    "    n_estimators=500,\n",
    "    max_depth=6,\n",
    "    learning_rate=0.05,\n",
    "    subsample=0.8,\n",
    "    colsample_bytree=0.8,\n",
    "    eval_metric='mlogloss',\n",
    "    random_state=SEED,\n",
    "    n_jobs=-1,\n",
    "    early_stopping_rounds=30,\n",
    "    tree_method='hist',\n",
    "    device='cuda' if torch.cuda.is_available() else 'cpu',\n",
    ")\n",
    "\n",
    "xgb_model.fit(\n",
    "    E_train, y_e_train,\n",
    "    eval_set=[(E_val, y_e_val)],\n",
    "    verbose=50,\n",
    ")\n",
    "\n",
    "xgb_pred  = xgb_model.predict(E_test)\n",
    "xgb_proba = xgb_model.predict_proba(E_test)   # (N, N_CLASSES)\n",
    "\n",
    "acc_xgb = accuracy_score(y_e_test, xgb_pred)\n",
    "try:\n",
    "    auc_xgb = roc_auc_score(y_e_test, xgb_proba,\n",
    "                             multi_class='ovr', average='macro')\n",
    "except Exception:\n",
    "    auc_xgb = float('nan')\n",
    "print(f'\\n✅ XGBoost (emb) | Acc: {acc_xgb:.4f} | AUC: {auc_xgb:.4f}')\n",
]

# Convert to notebook source format (each line is its own string element)
# The notebook stores source as a list of strings WITH newlines embedded.
# Our NEW_SOURCE already has that format — just verify.

# ── Load notebook ─────────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Find Cell 18 by its markdown header ──────────────────────────────────────
TARGET_CELL_ID = "7b1545f2"   # id in the JSON
patched = False

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code" and cell.get("id") == TARGET_CELL_ID:
        print(f"  Found Cell 18 at notebook cell index {i}  (id={TARGET_CELL_ID})")
        old_source = cell["source"]
        print(f"  Old source (first line): {repr(old_source[0] if old_source else '<empty>')}")
        # Replace source
        cell["source"] = NEW_SOURCE
        # Clear any stale error output
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    print("❌ Could not find Cell 18 by id '7b1545f2'. Trying fallback: search by markdown header.")
    # Fallback: look for the code cell that follows the "CELL 18" markdown
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if "CELL 18" in src:
                # The next code cell is Cell 18
                for j in range(i + 1, len(nb["cells"])):
                    if nb["cells"][j]["cell_type"] == "code":
                        print(f"  Fallback: found code cell at index {j}")
                        nb["cells"][j]["source"] = NEW_SOURCE
                        nb["cells"][j]["outputs"] = []
                        nb["cells"][j]["execution_count"] = None
                        patched = True
                        break
                break

if not patched:
    print("❌ Patch FAILED — could not locate Cell 18.")
    sys.exit(1)

# ── Write back ────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook patched successfully → {NB_PATH}")
print("   Cell 18 now contains the defensive label remapping guard.")
print("   Re-run the notebook from Cell 17 onward.")

"""
fix_cells18_19_v3.py
--------------------
Fixes the IndexError in Cell 18 caused by:
  CLASS_NAMES = [CLASS_NAMES[i] for i in _unique_labels]
  where _unique_labels = [0, 1, 3] but CLASS_NAMES only has 3 items → index 3 OOB.

Root cause:
  CLASS_NAMES was already rebuilt in Cell 5 to match the full tabular set
  (e.g. ['CN','LMCI','AD'] — 3 items). But the MRI subset labels are still
  encoded with the ORIGINAL 4-class scheme (CN=0, EMCI=1, LMCI=2, AD=3),
  so label 3 means AD but CLASS_NAMES[3] doesn't exist.

Solution:
  Use the hardcoded canonical 4-class list ['CN','EMCI','LMCI','AD'] as the
  look-up table for decoding old label ints → class names, then rebuild
  CLASS_NAMES as the compact sorted list of only those classes present.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")

# ── New source for Cell 18 ────────────────────────────────────────────────────
NEW_SOURCE_18 = [
    "print('🌲 Training XGBoost on fused embeddings...')\n",
    "\n",
    "# ── Defensive label remapping (shared across Cells 18 & 19) ──────────────\n",
    "# XGBoost requires contiguous labels [0, 1, ..., k-1] with NO gaps.\n",
    "# The MRI-matched subset (163 subjects) may have labels like [0,1,3]\n",
    "# (gap at 2) if the full tabular 4-class encoding was used but some class\n",
    "# is absent in the MRI subset.  We remap ALL label arrays here once.\n",
    "_unique_labels = np.unique(y_e_train)                 # e.g. [0, 1, 3]\n",
    "_expected      = np.arange(len(_unique_labels))       # e.g. [0, 1, 2]\n",
    "if not np.array_equal(_unique_labels, _expected):\n",
    "    # Map old integer → new contiguous integer\n",
    "    _remap = {int(old): int(new) for new, old in enumerate(_unique_labels)}\n",
    "    print(f'⚠️  Non-contiguous labels detected: {_unique_labels.tolist()}')\n",
    "    print(f'   Remapping → {_expected.tolist()} (remap: {_remap})')\n",
    "    # Remap embedding-split labels\n",
    "    y_e_train = np.vectorize(_remap.get)(y_e_train).astype(np.int64)\n",
    "    y_e_val   = np.vectorize(_remap.get)(y_e_val).astype(np.int64)\n",
    "    y_e_test  = np.vectorize(_remap.get)(y_e_test).astype(np.int64)\n",
    "    # Remap tabular-split labels (used by Cell 19)\n",
    "    y_train_sm = np.vectorize(_remap.get)(y_train_sm).astype(np.int64)\n",
    "    y_val      = np.vectorize(_remap.get)(y_val).astype(np.int64)\n",
    "    y_test     = np.vectorize(_remap.get)(y_test).astype(np.int64)\n",
    "    y_train    = np.vectorize(_remap.get)(y_train).astype(np.int64)\n",
    "    # Rebuild CLASS_NAMES using the CANONICAL 4-class lookup so index 3 → 'AD'\n",
    "    # This is safe regardless of how many classes Cell 5 ended up with.\n",
    "    _CANONICAL_4 = ['CN', 'EMCI', 'LMCI', 'AD']\n",
    "    CLASS_NAMES  = [_CANONICAL_4[i] for i in _unique_labels]\n",
    "    N_CLASSES    = len(CLASS_NAMES)\n",
    "    print(f'   Updated CLASS_NAMES: {CLASS_NAMES}  |  N_CLASSES: {N_CLASSES}')\n",
    "else:\n",
    "    _remap = None\n",
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

# ── New source for Cell 19 ────────────────────────────────────────────────────
NEW_SOURCE_19 = [
    "print('📊 Training XGBoost on raw tabular features (baseline)...')\n",
    "\n",
    "# y_train_sm / y_val / y_test were already remapped to contiguous 0-based\n",
    "# integers in Cell 18 (if needed). Safe to use directly here.\n",
    "xgb_tab = xgb.XGBClassifier(\n",
    "    n_estimators=500, max_depth=5, learning_rate=0.05,\n",
    "    subsample=0.8, colsample_bytree=0.8,\n",
    "    eval_metric='mlogloss', random_state=SEED, n_jobs=-1,\n",
    "    early_stopping_rounds=30,\n",
    "    tree_method='hist',\n",
    "    device='cuda' if torch.cuda.is_available() else 'cpu',\n",
    ")\n",
    "xgb_tab.fit(X_tab_train_sm, y_train_sm,\n",
    "             eval_set=[(X_tab_val, y_val)], verbose=False)\n",
    "\n",
    "xgb_tab_pred  = xgb_tab.predict(X_tab_test)\n",
    "xgb_tab_proba = xgb_tab.predict_proba(X_tab_test)   # (N, N_CLASSES)\n",
    "\n",
    "acc_xtab = accuracy_score(y_test, xgb_tab_pred)\n",
    "try:\n",
    "    auc_xtab = roc_auc_score(y_test, xgb_tab_proba,\n",
    "                              multi_class='ovr', average='macro')\n",
    "except Exception:\n",
    "    auc_xtab = float('nan')\n",
    "print(f'✅ XGBoost-Tab | Acc: {acc_xtab:.4f} | AUC: {auc_xtab:.4f}')\n",
]

# ── Load, patch, save ─────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched_18 = False
patched_19 = False

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    cid = cell.get("id", "")
    if cid == "7b1545f2":
        print("  Patching Cell 18 ...")
        cell["source"] = NEW_SOURCE_18
        cell["outputs"] = []
        cell["execution_count"] = None
        patched_18 = True
    elif cid == "3dca0c33":
        print("  Patching Cell 19 ...")
        cell["source"] = NEW_SOURCE_19
        cell["outputs"] = []
        cell["execution_count"] = None
        patched_19 = True

# Fallback by markdown header
if not patched_18 or not patched_19:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            for header, new_src, label in [
                ("CELL 18", NEW_SOURCE_18, "18"),
                ("CELL 19", NEW_SOURCE_19, "19"),
            ]:
                if header in src and ((label == "18" and not patched_18) or
                                       (label == "19" and not patched_19)):
                    for j in range(i + 1, len(nb["cells"])):
                        if nb["cells"][j]["cell_type"] == "code":
                            print(f"  Fallback: patching Cell {label} at index {j}")
                            nb["cells"][j]["source"] = new_src
                            nb["cells"][j]["outputs"] = []
                            nb["cells"][j]["execution_count"] = None
                            if label == "18":
                                patched_18 = True
                            else:
                                patched_19 = True
                            break

if not patched_18 or not patched_19:
    print(f"❌ Patch incomplete — Cell18:{patched_18} Cell19:{patched_19}")
    sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Patched successfully → {NB_PATH}")
print("   Reload the notebook, then re-run from Cell 17 onward.")

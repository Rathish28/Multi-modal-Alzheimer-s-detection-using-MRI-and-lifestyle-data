"""
fix_cells18_19.py
-----------------
Patches Cell 18 and Cell 19 in alzheimer_multimodal_GPU.ipynb.

Root cause:
  The 163 MRI-matched subjects end up with labels [0,1,3] (gap at 2) because
  the full tabular dataset has 4 classes (CN/EMCI/LMCI/AD) but the MRI subset
  only contains CN/EMCI/AD. XGBoost requires [0,1,...,k-1] with no gaps.

Strategy:
  - Cell 18 fix (embedding-based XGBoost): already patched, remap y_e_* splits.
  - Cell 19 fix (raw-tabular XGBoost): remap y_train_sm / y_val / y_test
    using the same _remap dict so all cells stay consistent.

The remap dict (_remap) is shared — it is built once in Cell 18 and reused
in Cell 19, keeping everything deterministic.
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
    "# The MRI-matched subset (163 subjects) may have a different class\n",
    "# distribution than the full tabular table (2409 subjects).\n",
    "# E.g. full table: CN/LMCI/AD → [0,1,2]; but MRI subjects: CN/EMCI/AD\n",
    "# with 4-class encoding → labels [0,1,3].  XGBoost cannot handle the gap.\n",
    "# We detect and fix this ONCE here, and store _remap for Cell 19 to reuse.\n",
    "_unique_labels = np.unique(y_e_train)\n",
    "_expected      = np.arange(len(_unique_labels))\n",
    "if not np.array_equal(_unique_labels, _expected):\n",
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
    "    # Update CLASS_NAMES / N_CLASSES to match MRI-subset classes only\n",
    "    CLASS_NAMES = [CLASS_NAMES[i] for i in _unique_labels]\n",
    "    N_CLASSES   = len(CLASS_NAMES)\n",
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

# ── Load notebook ─────────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched_18 = False
patched_19 = False

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue

    cid = cell.get("id", "")
    if cid == "7b1545f2":           # Cell 18
        print(f"  Patching Cell 18 at index {i}")
        cell["source"] = NEW_SOURCE_18
        cell["outputs"] = []
        cell["execution_count"] = None
        patched_18 = True

    elif cid == "3dca0c33":         # Cell 19
        print(f"  Patching Cell 19 at index {i}")
        cell["source"] = NEW_SOURCE_19
        cell["outputs"] = []
        cell["execution_count"] = None
        patched_19 = True

# Fallback: locate by markdown header if IDs changed
if not patched_18 or not patched_19:
    print("  Falling back to markdown-header search...")
    looking_for = []
    if not patched_18:
        looking_for.append(("CELL 18", NEW_SOURCE_18, "18"))
    if not patched_19:
        looking_for.append(("CELL 19", NEW_SOURCE_19, "19"))

    for header, new_src, label in looking_for:
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if header in src:
                    for j in range(i + 1, len(nb["cells"])):
                        if nb["cells"][j]["cell_type"] == "code":
                            print(f"  Found Cell {label} via header at index {j}")
                            nb["cells"][j]["source"] = new_src
                            nb["cells"][j]["outputs"] = []
                            nb["cells"][j]["execution_count"] = None
                            if label == "18":
                                patched_18 = True
                            else:
                                patched_19 = True
                            break
                    break

if not patched_18:
    print("❌ Could not locate Cell 18 — patch FAILED.")
    sys.exit(1)
if not patched_19:
    print("❌ Could not locate Cell 19 — patch FAILED.")
    sys.exit(1)

# ── Write back ────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Both cells patched → {NB_PATH}")
print("   Cell 18: remaps ALL label arrays (embedding + tabular splits)")
print("   Cell 19: uses already-remapped labels — no XGBoost gap error")
print("\n   Reload the notebook, then re-run from Cell 17 onward.")

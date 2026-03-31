"""
fix_cell21.py
-------------
Patches Cell 21 (Fusion Model Test Evaluation) to fix:
  IndexError: index 3 is out of bounds for axis 1 with size 3

Root cause:
  y_test_eval is collected from test_loader which still carries the ORIGINAL
  label values (e.g. 3 for AD in 4-class encoding). After Cell 18 remapped
  y_e_* arrays to [0,1,2], fusion_proba has shape (N, 3). But valid_cls from
  y_test_eval contains [0,1,3] so fusion_proba[:, 3] crashes.

Fix:
  After collecting y_test_eval from the DataLoader, remap it using _remap
  (which Cell 18 stored as a module-level variable). If _remap is None
  (labels were already contiguous), nothing changes.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")
CELL_ID  = "165646a2"   # Cell 21

NEW_SOURCE_21 = [
    "print('📊 Evaluating Fusion Model on test set...')\n",
    "\n",
    "fusion_model.eval()\n",
    "all_logits, all_labels = [], []\n",
    "\n",
    "with torch.no_grad(), autocast(enabled=USE_AMP):\n",
    "    for mri_b, tab_b, lbl_b in test_loader:\n",
    "        mri_b = mri_b.to(DEVICE, non_blocking=True)\n",
    "        tab_b = tab_b.to(DEVICE, non_blocking=True)\n",
    "        logits = fusion_model(mri_b, tab_b)\n",
    "        all_logits.append(F.softmax(logits, dim=1).cpu().float().numpy())\n",
    "        all_labels.extend(lbl_b.numpy())\n",
    "\n",
    "fusion_proba = np.concatenate(all_logits)\n",
    "fusion_pred  = fusion_proba.argmax(1)\n",
    "y_test_eval  = np.array(all_labels, dtype=np.int64)\n",
    "\n",
    "# Remap y_test_eval to contiguous [0,1,...,k-1] if Cell 18 applied a remap.\n",
    "# _remap is set in Cell 18: dict like {0:0, 1:1, 3:2} or None if no remap needed.\n",
    "if '_remap' in dir() and _remap is not None:\n",
    "    y_test_eval = np.vectorize(_remap.get)(y_test_eval).astype(np.int64)\n",
    "\n",
    "acc_fus = accuracy_score(y_test_eval, fusion_pred)\n",
    "try:\n",
    "    auc_fus = roc_auc_score(\n",
    "        y_test_eval,\n",
    "        fusion_proba,\n",
    "        multi_class='ovr',\n",
    "        average='macro'\n",
    "    )\n",
    "except Exception:\n",
    "    auc_fus = float('nan')\n",
    "print(f'✅ Fusion Model | Acc: {acc_fus:.4f} | AUC: {auc_fus:.4f}')\n",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and cell.get("id") == CELL_ID:
        print(f"  Patching Cell 21 (id={CELL_ID}) ...")
        cell["source"] = NEW_SOURCE_21
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    print("  Fallback: searching by markdown header 'CELL 21' ...")
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown":
            if "CELL 21" in "".join(cell.get("source", [])):
                for j in range(i + 1, len(nb["cells"])):
                    if nb["cells"][j]["cell_type"] == "code":
                        print(f"  Found at index {j}")
                        nb["cells"][j]["source"] = NEW_SOURCE_21
                        nb["cells"][j]["outputs"] = []
                        nb["cells"][j]["execution_count"] = None
                        patched = True
                        break
                break

if not patched:
    print("❌ Could not locate Cell 21."); sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Cell 21 patched → {NB_PATH}")
print("   Reload the notebook and re-run Cell 21.")

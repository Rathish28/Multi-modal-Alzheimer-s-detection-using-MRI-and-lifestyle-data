"""
fix_cells21_23.py
-----------------
Patches Cell 21 and Cell 23 together to fix their IndexError issues.

Cell 21 fix: Use 'globals()' instead of 'dir()' to check for _remap,
             which is more reliable in Jupyter notebook scope.

Cell 23 fix: Use col_i (loop counter) instead of cls_idx (label int) to
             index into CLASS_NAMES — safe regardless of label gaps.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")

# ── Cell 21 — Fusion Model Test Evaluation ───────────────────────────────────
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
    "# Remap y_test_eval to contiguous [0,1,...,k-1] using _remap from Cell 18.\n",
    "# Use globals() which is reliable in Jupyter notebook scope.\n",
    "_g = globals()\n",
    "if '_remap' in _g and _g['_remap'] is not None:\n",
    "    y_test_eval = np.vectorize(_g['_remap'].get)(y_test_eval).astype(np.int64)\n",
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

# ── Cell 23 — Confusion Matrices & ROC Curves ────────────────────────────────
NEW_SOURCE_23 = [
    "from sklearn.preprocessing import label_binarize\n",
    "from sklearn.metrics import roc_curve, auc as sklearn_auc\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
    "\n",
    "# ── Confusion matrix ──────────────────────────────────────────────────────\n",
    "cm = confusion_matrix(y_test_eval, ens_pred, labels=list(range(N_CLASSES)))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0],\n",
    "            annot_kws={'size': 13})\n",
    "axes[0].set(xlabel='Predicted', ylabel='True',\n",
    "            title='Ensemble — Confusion Matrix')\n",
    "\n",
    "# ── ROC curves ────────────────────────────────────────────────────────────\n",
    "# y_test_eval is remapped to [0,1,...,N_CLASSES-1] by Cell 21.\n",
    "# Iterate by col_i (0-based counter) for CLASS_NAMES and proba columns.\n",
    "classes_in_test = sorted(np.unique(y_test_eval))   # e.g. [0, 1, 2]\n",
    "y_bin  = label_binarize(y_test_eval, classes=classes_in_test)\n",
    "colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']\n",
    "\n",
    "for col_i, cls_idx in enumerate(classes_in_test):\n",
    "    # col_i  is the binarized column AND the CLASS_NAMES index (always in-bounds)\n",
    "    # cls_idx is the actual label integer in y_test_eval (same after remap)\n",
    "    cls_name = CLASS_NAMES[col_i]\n",
    "    fpr, tpr, _ = roc_curve(y_bin[:, col_i], ens_proba[:, cls_idx])\n",
    "    rval         = sklearn_auc(fpr, tpr)\n",
    "    axes[1].plot(fpr, tpr, color=colors[col_i % 4], lw=2,\n",
    "                 label=f'{cls_name} (AUC={rval:.3f})')\n",
    "\n",
    "axes[1].plot([0,1],[0,1],'k--',lw=1)\n",
    "axes[1].set(xlabel='False Positive Rate', ylabel='True Positive Rate',\n",
    "            title='ROC Curves (One-vs-Rest)')\n",
    "axes[1].legend(loc='lower right', fontsize=11)\n",
    "axes[1].grid(True, alpha=0.3)\n",
    "\n",
    "fig.tight_layout()\n",
    "fig.savefig(PLOT_DIR / 'evaluation_plots.png', dpi=150)\n",
    "plt.close(fig)\n",
    "print(f'✅ Saved: {PLOT_DIR / \"evaluation_plots.png\"}')\n",
]

# ── Load, patch, save ─────────────────────────────────────────────────────────
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

PATCHES = {
    "165646a2": ("21", NEW_SOURCE_21),
    "3bf39172": ("23", NEW_SOURCE_23),
}
patched = {}

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    cid = cell.get("id", "")
    if cid in PATCHES:
        num, src = PATCHES[cid]
        print(f"  Patching Cell {num} (id={cid}) ...")
        cell["source"] = src
        cell["outputs"] = []
        cell["execution_count"] = None
        patched[cid] = True

# Fallback by header
for cid, (num, src) in PATCHES.items():
    if cid not in patched:
        header = f"CELL {num}"
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "markdown" and header in "".join(cell.get("source", [])):
                for j in range(i + 1, len(nb["cells"])):
                    if nb["cells"][j]["cell_type"] == "code":
                        print(f"  Fallback: Cell {num} at index {j}")
                        nb["cells"][j]["source"] = src
                        nb["cells"][j]["outputs"] = []
                        nb["cells"][j]["execution_count"] = None
                        patched[cid] = True
                        break
                break

missing = [PATCHES[c][0] for c in PATCHES if c not in patched]
if missing:
    print(f"❌ Could not patch: Cell(s) {missing}"); sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Cells 21 & 23 patched → {NB_PATH}")
print("   Reload the notebook and re-run from Cell 21 onward.")

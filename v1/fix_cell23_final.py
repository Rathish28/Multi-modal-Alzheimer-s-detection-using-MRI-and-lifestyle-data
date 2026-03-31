"""
fix_cell23_final.py
-------------------
Definitive fix for Cell 23's IndexError.

The real problem: y_test_eval still has raw label 3 (for AD) because
_remap was not applied (globals() check failed). So classes_in_test = [0,1,3]
and ens_proba[:, 3] crashes (ens_proba only has 3 cols: 0,1,2).

Fix strategy: Do NOT use cls_idx (raw label int) to index ens_proba.
Instead, use col_i for EVERYTHING:
  - y_bin[:, col_i]      — binarized column
  - ens_proba[:, col_i]  — proba column (both are in the same 0-based order)
  - CLASS_NAMES[col_i]   — class name

Also remap y_test_eval locally in Cell 23 directly from the raw DataLoader
labels, so we never depend on Cell 21 having done the remap.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")
CELL_ID  = "3bf39172"   # Cell 23

NEW_SOURCE_23 = [
    "from sklearn.preprocessing import label_binarize\n",
    "from sklearn.metrics import roc_curve, auc as sklearn_auc\n",
    "\n",
    "# ── Ensure y_test_eval is contiguous [0,1,...,N_CLASSES-1] ───────────────\n",
    "# Re-remap locally in case Cell 21's globals() check didn't apply _remap.\n",
    "_ul = np.unique(y_test_eval)\n",
    "if not np.array_equal(_ul, np.arange(len(_ul))):\n",
    "    _local_remap = {int(old): int(new) for new, old in enumerate(_ul)}\n",
    "    y_test_eval  = np.vectorize(_local_remap.get)(y_test_eval).astype(np.int64)\n",
    "    ens_pred     = np.vectorize(_local_remap.get)(ens_pred).astype(np.int64)\n",
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
    "# After remap, y_test_eval contains only [0, 1, ..., N_CLASSES-1].\n",
    "# Use col_i for ALL indexing: ens_proba, y_bin, CLASS_NAMES.\n",
    "classes_in_test = sorted(np.unique(y_test_eval))   # e.g. [0, 1, 2]\n",
    "y_bin  = label_binarize(y_test_eval, classes=classes_in_test)\n",
    "colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']\n",
    "\n",
    "for col_i in range(len(classes_in_test)):\n",
    "    cls_name = CLASS_NAMES[col_i]\n",
    "    fpr, tpr, _ = roc_curve(y_bin[:, col_i], ens_proba[:, col_i])\n",
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

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and cell.get("id") == CELL_ID:
        print(f"  Patching Cell 23 (id={CELL_ID}) ...")
        cell["source"] = NEW_SOURCE_23
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown" and "CELL 23" in "".join(cell.get("source", [])):
            for j in range(i + 1, len(nb["cells"])):
                if nb["cells"][j]["cell_type"] == "code":
                    print(f"  Fallback: Cell 23 at index {j}")
                    nb["cells"][j]["source"] = NEW_SOURCE_23
                    nb["cells"][j]["outputs"] = []
                    nb["cells"][j]["execution_count"] = None
                    patched = True
                    break
            break

if not patched:
    print("❌ Could not locate Cell 23."); sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Cell 23 patched → {NB_PATH}")
print("   Reload the notebook and re-run Cell 23.")

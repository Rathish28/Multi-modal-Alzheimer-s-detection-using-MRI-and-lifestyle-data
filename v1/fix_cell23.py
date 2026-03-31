"""
fix_cell23.py
-------------
Patches Cell 23 (Confusion Matrices & ROC Curves) to fix:
  IndexError: list index out of range
  at: cls_name = CLASS_NAMES[cls_idx]

Root cause:
  classes_in_test uses the label integers from y_test_eval as indices into
  CLASS_NAMES. If labels are [0,1,3] (non-contiguous), CLASS_NAMES[3] fails.
  Even after Cell 21's remap, the loop uses cls_idx as both:
    - a column index into ens_proba (correct)
    - an index into CLASS_NAMES (WRONG if labels were e.g. [0,1,3])

Fix:
  Enumerate CLASS_NAMES directly — use col_i as the class name index and
  cls_idx as the proba column index. This is always correct regardless of
  what integer labels are in y_test_eval.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")
CELL_ID  = "3bf39172"   # Cell 23

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
    "# y_test_eval is already remapped to [0,1,...,N_CLASSES-1] by Cell 21.\n",
    "# classes_in_test are the contiguous label ints present in the test set.\n",
    "classes_in_test = sorted(np.unique(y_test_eval))\n",
    "y_bin  = label_binarize(y_test_eval, classes=classes_in_test)\n",
    "colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']\n",
    "\n",
    "# col_i  = binary column index (0, 1, 2, ...)\n",
    "# cls_idx = actual label integer in y_test_eval (same as col_i after remap)\n",
    "# Use col_i to index CLASS_NAMES — guaranteed in-bounds after remap.\n",
    "for col_i, cls_idx in enumerate(classes_in_test):\n",
    "    cls_name = CLASS_NAMES[col_i]           # safe: col_i < N_CLASSES always\n",
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
    print("  Fallback: searching by markdown header ...")
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "markdown":
            if "CELL 23" in "".join(cell.get("source", [])):
                for j in range(i + 1, len(nb["cells"])):
                    if nb["cells"][j]["cell_type"] == "code":
                        print(f"  Found at index {j}")
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

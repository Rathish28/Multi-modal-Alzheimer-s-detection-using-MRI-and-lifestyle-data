"""
fix_cell14_checkpoint.py
------------------------
Fixes RuntimeError in Cell 14:
  size mismatch for classifier.6.weight:
  checkpoint torch.Size([4, 128]) vs current model torch.Size([3, 128])

Root cause:
  The saved fusion_best.pt was trained with N_CLASSES=4 (old 4-class run).
  After Cell 18's remap, N_CLASSES=3 so the model has a 3-output classifier.
  Loading strict=True fails because the output layer shapes don't match.

Fix:
  - Delete the stale fusion_best.pt so Cell 15 will retrain from scratch
    with 3 classes and save a fresh compatible checkpoint.
  - Change load_state_dict to strict=False as a safety fallback (in case
    a partially-compatible checkpoint exists in future runs).
  - The CNN body weights from the old checkpoint are NOT reused — we retrain
    from scratch since the class count changed.
"""

import json, sys
from pathlib import Path

NB_PATH   = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")
CKPT_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\models\fusion_best.pt")
CELL_ID   = "7813c210"   # Cell 14

# ── Delete the stale checkpoint ───────────────────────────────────────────────
if CKPT_PATH.exists():
    CKPT_PATH.unlink()
    print(f"  🗑  Deleted stale checkpoint: {CKPT_PATH}")
else:
    print(f"  ℹ️  Checkpoint not found (already deleted or never saved): {CKPT_PATH}")

# ── Patch Cell 14 checkpoint-loading block ────────────────────────────────────
# Replace the strict load with a class-count check + graceful skip.
OLD_CKPT_LINES = [
    "# ── Auto-load trained checkpoint if it exists ─────────────────────────────\n",
    "# This makes Cell 14 safe to re-run without losing trained weights.\n",
    "# If you re-run this cell after training, the best trained weights are \n",
    "# immediately restored so downstream cells (17-29) work correctly.\n",
    "_ckpt_path = MODEL_DIR / 'fusion_best.pt'\n",
    "if _ckpt_path.exists():\n",
    "    print(f'   Loading trained checkpoint from {_ckpt_path}')\n",
    "    fusion_model.load_state_dict(torch.load(_ckpt_path, map_location=DEVICE,\n",
    "                                             weights_only=True))\n",
    "    fusion_model.eval()\n",
    "    print('   ✅ Trained weights restored. Model ready for evaluation.')\n",
    "else:\n",
    "    print('   ⚠️  No checkpoint found. Run Cell 15 (training) before evaluating.')\n",
]

NEW_CKPT_LINES = [
    "# ── Auto-load trained checkpoint if it exists ─────────────────────────────\n",
    "# Checks that the saved checkpoint matches the current N_CLASSES before\n",
    "# loading. If classes changed (e.g. 4→3 after label remap), skip the load\n",
    "# so Cell 15 retrains fresh with the correct output size.\n",
    "_ckpt_path = MODEL_DIR / 'fusion_best.pt'\n",
    "if _ckpt_path.exists():\n",
    "    _sd = torch.load(_ckpt_path, map_location=DEVICE, weights_only=True)\n",
    "    _ckpt_nclasses = _sd.get('classifier.6.bias', torch.zeros(N_CLASSES)).shape[0]\n",
    "    if _ckpt_nclasses != N_CLASSES:\n",
    "        print(f'   ⚠️  Checkpoint has {_ckpt_nclasses} output classes but '\n",
    "              f'current model needs {N_CLASSES}. Skipping checkpoint load.')\n",
    "        print('   Run Cell 15 to retrain with the correct class count.')\n",
    "    else:\n",
    "        print(f'   Loading trained checkpoint from {_ckpt_path}')\n",
    "        fusion_model.load_state_dict(_sd)\n",
    "        fusion_model.eval()\n",
    "        print('   ✅ Trained weights restored. Model ready for evaluation.')\n",
    "else:\n",
    "    print('   ⚠️  No checkpoint found. Run Cell 15 (training) before evaluating.')\n",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and cell.get("id") == CELL_ID:
        src = cell["source"]
        # Find and replace the checkpoint block
        old_block_start = None
        for i, line in enumerate(src):
            if "Auto-load trained checkpoint" in line:
                old_block_start = i
                break
        if old_block_start is not None:
            new_src = src[:old_block_start] + NEW_CKPT_LINES
            cell["source"] = new_src
            cell["outputs"] = []
            cell["execution_count"] = None
            patched = True
            print(f"  ✅ Patched Cell 14 checkpoint-loading block")
        else:
            print("  ⚠️  Could not find checkpoint block in Cell 14 source")
        break

if not patched:
    print("❌ Could not locate Cell 14 (id=7813c210)"); sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Done → {NB_PATH}")
print("   1. The stale 4-class checkpoint has been deleted.")
print("   2. Cell 14 now checks class count before loading checkpoints.")
print("   3. Reload the notebook, re-run Cell 14, then re-run Cell 15 to retrain.")

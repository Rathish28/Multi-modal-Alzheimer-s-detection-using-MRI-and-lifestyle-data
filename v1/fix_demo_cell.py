"""
fix_demo_cell.py
----------------
Fixes SyntaxError: invalid character '─' (U+2500)
in the demo patients cell (cell id 88469975).

The f-string f'\n{'─'*55}' uses a box-drawing character inside a
curly-brace expression, which Python cannot parse as an operator.
Replace with regular ASCII '-'.
"""

import json, sys
from pathlib import Path

NB_PATH = Path(r"C:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb")
CELL_ID  = "88469975"

NEW_SOURCE = [
    "# Three contrasting patient profiles — run this to demonstrate to your mentor\n",
    "DEMO_PATIENTS = {\n",
    "    'Patient A (CN - Low Risk)': {\n",
    "        'AGE':68,'APOE4':0,'MMSE':29,'CDRSB':0.0,'ADAS11':8.0,'ADAS13':11.0,\n",
    "        'FAQ':0,'RAVLT_IMMEDIATE':48,'RAVLT_LEARNING':6,'LDELTOTAL':10,\n",
    "        'HIPPOCAMPUS':7800,'ENTORHINAL':3900,'VENTRICLES':25000,\n",
    "        'VSBPSYS':120,'VSBPDIA':78,'VSWEIGHT':72,\n",
    "    },\n",
    "    'Patient B (EMCI - Moderate Risk)': {\n",
    "        'AGE':74,'APOE4':1,'MMSE':26,'CDRSB':1.5,'ADAS11':14.0,'ADAS13':18.0,\n",
    "        'FAQ':3,'RAVLT_IMMEDIATE':32,'RAVLT_LEARNING':3,'LDELTOTAL':6,\n",
    "        'HIPPOCAMPUS':6800,'ENTORHINAL':3200,'VENTRICLES':33000,\n",
    "        'VSBPSYS':138,'VSBPDIA':88,'VSWEIGHT':78,\n",
    "    },\n",
    "    'Patient C (AD - High Risk)': {\n",
    "        'AGE':82,'APOE4':2,'MMSE':19,'CDRSB':5.5,'ADAS11':28.0,'ADAS13':36.0,\n",
    "        'FAQ':18,'RAVLT_IMMEDIATE':18,'RAVLT_LEARNING':1,'LDELTOTAL':2,\n",
    "        'HIPPOCAMPUS':5200,'ENTORHINAL':2400,'VENTRICLES':55000,\n",
    "        'VSBPSYS':155,'VSBPDIA':95,'VSWEIGHT':61,\n",
    "    },\n",
    "}\n",
    "\n",
    "demo_results = {}\n",
    "for pname, vals in DEMO_PATIENTS.items():\n",
    "    print('\\n' + '-'*55)\n",
    "    print(f'🩺 {pname}')\n",
    "    demo_results[pname] = predict_patient(None, vals)\n",
    "\n",
    "# Summary table\n",
    "print('\\n' + '='*60)\n",
    "print('  DEMO SUMMARY')\n",
    "print('='*60)\n",
    "print(f'  {\"Patient\":<35} | {\"Diagnosis\":6} | {\"AD Risk %\"}')\n",
    "print('  ' + '-'*55)\n",
    "for pname, res in demo_results.items():\n",
    "    short = pname.split('(')[0].strip()\n",
    "    print(f'  {short:<35} | {res[\"Diagnosis\"]:6s}   | {res[\"AD_Risk_%\"]:>6.1f}%')\n",
    "print('='*60)\n",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and cell.get("id") == CELL_ID:
        print(f"  Patching demo cell (id={CELL_ID}) ...")
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        break

if not patched:
    # Fallback: find by content
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "DEMO_PATIENTS" in src and "predict_patient" in src:
                print(f"  Fallback: found demo cell at index {i}")
                cell["source"] = NEW_SOURCE
                cell["outputs"] = []
                cell["execution_count"] = None
                patched = True
                break

if not patched:
    print("❌ Could not locate the demo cell."); sys.exit(1)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Demo cell patched → {NB_PATH}")
print("   Reload the notebook and re-run the demo cell.")

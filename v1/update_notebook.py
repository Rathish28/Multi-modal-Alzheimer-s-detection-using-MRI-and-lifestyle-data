import json
import os

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 7 (index 7 or 8 based on execution_count and source)
cell_7_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if any('GPU Resize to target_shape' in line for line in source) or any('load_and_preprocess_mri' in line for line in source):
            cell_7_idx = i
            break

if cell_7_idx is not None:
    source = nb['cells'][cell_7_idx]['source']

    # check if process_all_mri is already present
    has_process = any('def process_all_mri' in line for line in source)

    if not has_process:
        # Prepend new code right before the final print statements
        new_code = [
            "\n",
            "def process_all_mri(mri_df: pd.DataFrame):\n",
            "    \"\"\"\n",
            "    Process all MRI NIfTI files listed in the requested DataFrame.\n",
            "    Leverages GPU for fast resampling if DEVICE is set to cuda.\n",
            "    \"\"\"\n",
            "    volumes = []\n",
            "    rids = []\n",
            "    \n",
            "    for _, row in tqdm(mri_df.iterrows(), total=len(mri_df), desc='Preprocessing MRIs'):\n",
            "        try:\n",
            "            vol = load_and_preprocess_mri(row['PATH'])\n",
            "            volumes.append(vol)\n",
            "            rids.append(row['RID'])\n",
            "        except Exception as e:\n",
            "            print(f'⚠️ Error on {row[\"PATH\"]}: {e}')\n",
            "            \n",
            "    return np.array(volumes, dtype=np.float32), np.array(rids, dtype=np.int64)\n",
            "\n",
            "\n"
        ]

        print_lines = []
        for i in reversed(range(len(source))):
            if source[i].startswith('print('):
                print_lines.insert(0, source.pop(i))
            else:
                if source[i].strip() == "":
                    source.pop(i)
                else:
                    break

        source.extend(new_code)
        source.extend(print_lines)
        nb['cells'][cell_7_idx]['source'] = source

        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

        print("Notebook updated successfully: added 'process_all_mri' to cell 7.")
    else:
        print("Function 'process_all_mri' is already in cell 7.")
else:
    print("Could not find cell 7. Please check manually.")

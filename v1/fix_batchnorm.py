import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        for j, line in enumerate(source):
            if "train_loader = DataLoader(" in line and "drop_last=True" not in source[j+1]:
                # We need to add drop_last=True to the train_loader parameters
                # The original is spread over two lines:
                # 1178: train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                # 1179:                           num_workers=NUM_WORKERS, pin_memory=PIN)
                if "pin_memory=PIN)" in source[j+1]:
                    source[j+1] = source[j+1].replace("pin_memory=PIN)", "pin_memory=PIN, drop_last=True)")
                    found = True

        if found:
            nb['cells'][i]['source'] = source
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully: Added drop_last=True to train_loader in Cell 11.")
else:
    print("Could not find the train_loader line. Please check manually.")

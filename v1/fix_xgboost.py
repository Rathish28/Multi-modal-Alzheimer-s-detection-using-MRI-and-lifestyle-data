import json

notebook_path = r"c:\Users\Rathish K\Documents\ML\v1\alzheimer_multimodal_GPU.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell.get('source', []))
    
    # Update Cell 18
    if "xgb_model.fit(" in source and "y_e_train_xgb" not in source:
        new_source = source.replace(
            "xgb_model.fit(\n    E_train, y_e_train,", 
            "y_e_train_xgb = np.where(y_e_train == 3, 2, y_e_train)\n"
            "y_e_val_xgb   = np.where(y_e_val == 3, 2, y_e_val)\n\n"
            "xgb_model.fit(\n    E_train, y_e_train_xgb,"
        )
        new_source = new_source.replace(
            "eval_set=[(E_val, y_e_val)],",
            "eval_set=[(E_val, y_e_val_xgb)],"
        )
        new_source = new_source.replace(
            "xgb_pred  = xgb_model.predict(E_test)\nxgb_proba = xgb_model.predict_proba(E_test)",
            "xgb_proba_3 = xgb_model.predict_proba(E_test)\n"
            "xgb_proba = np.zeros((len(E_test), 4))\n"
            "xgb_proba[:, :2] = xgb_proba_3[:, :2]\n"
            "xgb_proba[:, 3]  = xgb_proba_3[:, 2]\n"
            "xgb_pred  = xgb_proba.argmax(axis=1)"
        )
        nb['cells'][idx]['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

    # Update Cell 19
    elif "xgb_tab.fit(X_tab_train_sm, y_train_sm," in source and "y_train_sm_xgb" not in source:
        new_source = source.replace(
            "xgb_tab.fit(X_tab_train_sm, y_train_sm,\n             eval_set=[(X_tab_val, y_val)], verbose=False)",
            "y_train_sm_xgb = np.where(y_train_sm == 3, 2, y_train_sm)\n"
            "y_val_xgb      = np.where(y_val == 3, 2, y_val)\n"
            "xgb_tab.fit(X_tab_train_sm, y_train_sm_xgb,\n"
            "             eval_set=[(X_tab_val, y_val_xgb)], verbose=False)"
        )
        new_source = new_source.replace(
            "xgb_tab_pred  = xgb_tab.predict(X_tab_test)\nxgb_tab_proba = xgb_tab.predict_proba(X_tab_test)",
            "xgb_tab_proba_3 = xgb_tab.predict_proba(X_tab_test)\n"
            "xgb_tab_proba = np.zeros((len(X_tab_test), 4))\n"
            "xgb_tab_proba[:, :2] = xgb_tab_proba_3[:, :2]\n"
            "xgb_tab_proba[:, 3]  = xgb_tab_proba_3[:, 2]\n"
            "xgb_tab_pred  = xgb_tab_proba.argmax(axis=1)"
        )
        nb['cells'][idx]['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

    # Update Cell 29 -> prediction function
    elif "predict_patient(" in source and "xgb_p_3" not in source:
        new_source = source.replace(
            "xgb_p   = xgb_tab.predict_proba(tabular_features[np.newaxis, :])[0]",
            "xgb_p_3 = xgb_tab.predict_proba(tabular_features[np.newaxis, :])[0]\n"
            "        xgb_p   = np.array([xgb_p_3[0], xgb_p_3[1], 0.0, xgb_p_3[2]])"
        )
        nb['cells'][idx]['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("XGBoost class mismatch errors in Cells 18, 19, and 29 fixed!")

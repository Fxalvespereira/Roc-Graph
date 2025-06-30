import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix
import xlsxwriter

def load_z_values_from_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines if line.strip() and not line.lower().startswith('z-values')]

def add_threshold_column(df, z_value):
    df = df.copy()
    if "critical_value" in df.columns:
        df["threshold"] = df["critical_value"] * z_value
    else:
        df["threshold"] = None
    return df

def classify(df):
    concentration_col = [col for col in df.columns if 'Concentration' in col][0]
    intensity_col = [col for col in df.columns if 'Intensity' in col][0]
    threshold_col = "threshold"
    def lab(row):
        try:
            concentration = float(row[concentration_col])
            intensity = float(row[intensity_col])
            threshold = float(row[threshold_col])
        except (KeyError, ValueError, TypeError):
            return "Unclassified"
        if concentration != 0:
            return "True Positive" if intensity > threshold else "False Negative"
        else:
            return "False Positive" if intensity > threshold else "True Negative"
    df = df.copy()
    df["Classification"] = df.apply(lab, axis=1)
    df["True Label"] = df["Classification"].isin(["True Positive", "False Negative"]).astype(int)
    df["Above Threshold"] = df["Classification"].isin(["True Positive", "False Positive"]).astype(int)
    return df

def robust_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1,1):
        if np.all(y_true == 0):
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0,0]
    else:
        tn = fp = fn = tp = 0
    return tn, fp, fn, tp

def optimize_z_min_fp(sub, z_range, z_ref=5.5):
    concentration_col = [col for col in sub.columns if 'Concentration' in col][0]
    intensity_col = [col for col in sub.columns if 'Intensity' in col][0]
    cv = sub['critical_value'].iloc[0]
    y_true = (sub[concentration_col] != 0).astype(int)
    y_score = sub[intensity_col]

    z_candidates = []
    for z in z_range:
        threshold = cv * z
        preds = (y_score > threshold).astype(int)
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tp = ((preds == 1) & (y_true == 1)).sum()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        f1 = (2 * precision * recall) / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None
        z_candidates.append((z, tp, tn, fp, fn, acc, recall, specificity, precision, f1))

    # 1. Minimize FP
    min_fp = min(c[3] for c in z_candidates)
    fp_candidates = [c for c in z_candidates if c[3] == min_fp]
    # 2. Minimize FN
    min_fn = min(c[4] for c in fp_candidates)
    fn_candidates = [c for c in fp_candidates if c[4] == min_fn]
    # 3. Maximize accuracy
    max_acc = max(c[5] for c in fn_candidates if c[5] is not None)
    acc_candidates = [c for c in fn_candidates if c[5] == max_acc]
    # 4. Closest to z_ref
    best = min(acc_candidates, key=lambda x: abs(x[0] - z_ref))
    best_z, tp, tn, fp, fn, acc, recall, specificity, precision, f1 = best
    return best_z, fn, (tp, tn, fp, fn, acc, recall, specificity, precision, f1)

def process_compound_excels(input_folder='input', output_folder='output', z_value=5.5):
    os.makedirs(output_folder, exist_ok=True)
    processed_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_excel(file_path, header=None)
            n_rows, n_cols = df.shape
            sample_id_col = df.iloc[4:, 0].reset_index(drop=True)
            sample_id_header = df.iloc[3, 0]
            file_data = []
            compound_count = 0
            for start_col in range(1, n_cols, 6):
                headers = df.iloc[3, start_col:start_col+6].tolist()
                if all(h is np.nan or str(h).strip() == "" for h in headers):
                    continue
                compound_count += 1
                title_cell = df.iloc[2, start_col]
                compound_title = str(title_cell).strip() if pd.notna(title_cell) and str(title_cell).strip() != "" else f"Compound {compound_count}"
                try:
                    cv_val = float(df.iloc[0, start_col + 1])
                except (ValueError, TypeError, IndexError):
                    cv_val = None
                data_block = df.iloc[4:, start_col:start_col+6]
                data_block.columns = headers
                data_block = data_block.reset_index(drop=True)
                data_block.insert(0, sample_id_header, sample_id_col)
                data_block = data_block[
                    data_block[sample_id_header].notnull() &
                    (data_block[sample_id_header].astype(str).str.strip() != "")
                ]
                compound_id = compound_title.replace(' ', '_')
                data_block.insert(0, "critical_value", cv_val)
                data_block.insert(0, "compound_title", compound_title)
                data_block.insert(0, "compound_id", compound_id)
                data_block = add_threshold_column(data_block, z_value)
                data_block = classify(data_block)
                file_data.append(data_block)
            if file_data:
                output_df = pd.concat(file_data, ignore_index=True)
                out_csv = os.path.join(
                    output_folder, 
                    os.path.splitext(filename)[0] + "_processed.csv"
                )
                output_df.to_csv(out_csv, index=False)
                processed_files.append(out_csv)
                print(f"Saved {out_csv}")
            else:
                print(f"No compounds found in {filename}")
    return processed_files

def save_analysis_excel(input_csv, output_excel, z_list, z_input=5.5):
    with pd.ExcelWriter(output_excel, engine='xlsxwriter',
                        engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
        df = pd.read_csv(input_csv)
        df.to_excel(writer, sheet_name='Data', index=False)
        worksheet = writer.sheets['Data']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_len + 2)
        compounds = df['compound_id'].unique()
        summary_rows = []
        for compound in compounds:
            sub = df[df['compound_id'] == compound]
            compound_title = sub['compound_title'].iloc[0]
            cv = sub['critical_value'].iloc[0]
            concentration_col = [col for col in sub.columns if 'Concentration' in col][0]
            intensity_col = [col for col in sub.columns if 'Intensity' in col][0]
            y_true = (sub[concentration_col] != 0).astype(int)
            y_score = sub[intensity_col]
            filtered_zs = list(z_list)
            best_z, fn, stats = optimize_z_min_fp(sub, filtered_zs, z_input)
            tp, tn, fp, fn, acc, recall, specificity, precision, f1 = stats
            full_z_list = sorted(set(filtered_zs + [best_z, z_input]))
            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                for z in full_z_list:
                    summary_rows.append({
                        "Compound": compound_title,
                        "Critical Value": cv,
                        "Z_Type": "Sweep",
                        "Z_Value": z,
                        "Threshold": cv*z,
                        "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                        "Accuracy": None, "Sensitivity": None,
                        "Specificity": None, "Precision": None,
                        "F1 Score": None,
                        "NOTE": "Not enough classes for ROC/confusion"
                    })
                summary_rows.append({
                    "Compound": compound_title,
                    "Critical Value": cv,
                    "Z_Type": "Input Z",
                    "Z_Value": z_input,
                    "Threshold": cv*z_input,
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                    "Accuracy": None, "Sensitivity": None,
                    "Specificity": None, "Precision": None,
                    "F1 Score": None,
                    "NOTE": "Not enough classes for ROC/confusion"
                })
                summary_rows.append({
                    "Compound": compound_title,
                    "Critical Value": cv,
                    "Z_Type": "Optimal Z (Min FN, Max Correct)",
                    "Z_Value": 0,
                    "Threshold": 0,
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                    "Accuracy": None, "Sensitivity": None,
                    "Specificity": None, "Precision": None,
                    "F1 Score": None,
                    "NOTE": "Not enough classes for ROC/confusion"
                })
                continue
            for z in full_z_list:
                threshold = cv * z
                preds = (y_score > threshold).astype(int)
                tn = ((preds == 0) & (y_true == 0)).sum()
                fp = ((preds == 1) & (y_true == 0)).sum()
                fn = ((preds == 0) & (y_true == 1)).sum()
                tp = ((preds == 1) & (y_true == 1)).sum()
                total = tp + tn + fp + fn
                acc = (tp + tn) / total if total > 0 else None
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
                specificity = tn / (tn + fp) if (tn + fp) > 0 else None
                precision = tp / (tp + fp) if (tp + fp) > 0 else None
                f1 = (2 * precision * recall) / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None
                z_type = "Optimal Z" if np.isclose(z, best_z, atol=1e-10) else ("Input Z" if np.isclose(z, z_input, atol=1e-10) else "Sweep")
                summary_rows.append({
                    "Compound": compound_title,
                    "Critical Value": cv,
                    "Z_Type": z_type,
                    "Z_Value": z,
                    "Threshold": threshold,
                    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                    "Accuracy": acc, "Sensitivity": recall,
                    "Specificity": specificity, "Precision": precision,
                    "F1 Score": f1
                })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, index=False, sheet_name="Optimal_z_Summary")
        ws2 = writer.sheets["Optimal_z_Summary"]
        for i, col in enumerate(summary_df.columns):
            max_len = max(summary_df[col].astype(str).map(len).max(), len(col))
            ws2.set_column(i, i, max_len + 2)
        roc_ws = writer.book.add_worksheet('ROC_Curves')
        writer.sheets['ROC_Curves'] = roc_ws
        chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        row_cursor = 0
        for compound in compounds:
            sub = df[df['compound_id'] == compound]
            compound_title = sub['compound_title'].iloc[0]
            concentration_col = [col for col in sub.columns if 'Concentration' in col][0]
            intensity_col = [col for col in sub.columns if 'Intensity' in col][0]
            y_true = (sub[concentration_col] != 0).astype(int)
            y_score = sub[intensity_col]
            mask = (~pd.isna(y_score)) & (~pd.isna(y_true))
            y_true = y_true[mask]
            y_score = y_score[mask]
            cv = sub['critical_value'].iloc[0]
            filtered_zs = list(z_list)
            if len(set(y_true)) < 2:
                roc_ws.write(row_cursor, 0, f"{compound_title}: Only one class present, skipping ROC.")
                row_cursor += 2
                continue
            fpr, tpr, threshs = [], [], []
            for z in filtered_zs:
                threshold = cv * z
                preds = (y_score > threshold).astype(int)
                tn = ((preds == 0) & (y_true == 0)).sum()
                fp = ((preds == 1) & (y_true == 0)).sum()
                fn = ((preds == 0) & (y_true == 1)).sum()
                tp = ((preds == 1) & (y_true == 1)).sum()
                fpr_val = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                tpr_val = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                fpr.append(fpr_val)
                tpr.append(tpr_val)
                threshs.append(threshold)
            try:
                auc_val = auc([x for x in fpr if not np.isnan(x)],
                              [y for y in tpr if not np.isnan(y)])
            except Exception:
                auc_val = None
            roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': threshs, 'z': filtered_zs})
            roc_ws.write(row_cursor, 0, f"{compound_title} ROC (AUC={auc_val if auc_val is not None else 'N/A'})")
            for j, colname in enumerate(roc_df.columns):
                roc_ws.write(row_cursor+1, j, colname)
            for i, (_, row) in enumerate(roc_df.iterrows()):
                for j, val in enumerate(row):
                    roc_ws.write(row_cursor+2+i, j, val)
            chart.add_series({
                'name': f'{compound_title}',
                'categories': ['ROC_Curves', row_cursor+2, 0, row_cursor+2+len(fpr)-1, 0],
                'values':     ['ROC_Curves', row_cursor+2, 1, row_cursor+2+len(tpr)-1, 1],
                'marker': {'type': 'circle', 'size': 7},
                'line':   {'width': 2},
            })
            row_cursor += 3 + len(roc_df) + 1
        # Add 50% accuracy (chance) diagonal line
        chance_fpr_col = 25
        roc_ws.write(row_cursor+5, chance_fpr_col, "FPR_Chance")
        roc_ws.write(row_cursor+5, chance_fpr_col+1, "TPR_Chance")
        roc_ws.write(row_cursor+6, chance_fpr_col, 0)
        roc_ws.write(row_cursor+6, chance_fpr_col+1, 0)
        roc_ws.write(row_cursor+7, chance_fpr_col, 1)
        roc_ws.write(row_cursor+7, chance_fpr_col+1, 1)
        chart.add_series({
            'name': '50% Accuracy (Chance)',
            'categories': ['ROC_Curves', row_cursor+6, chance_fpr_col, row_cursor+7, chance_fpr_col],
            'values':     ['ROC_Curves', row_cursor+6, chance_fpr_col+1, row_cursor+7, chance_fpr_col+1],
            'marker': {'type': 'none'},
            'line': {'color': 'gray', 'dash_type': 'dash', 'width': 1.5},
        })
        chart.set_title({'name': 'All Compounds ROC Curves (User-defined z)'})
        chart.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_legend({'position': 'bottom'})
        roc_ws.insert_chart(row_cursor+2, 0, chart, {'x_scale': 2, 'y_scale': 2})
    print(f"Excel analysis file (optimal z, confusion, ROC, AUC) saved to {output_excel}")

def main():
    input_folder = "input"
    output_folder = "output"
    z_txt_file = "Z-Values to use.txt"
    z_list = load_z_values_from_txt(z_txt_file)
    z_value = 5.5
    print(f"Processing Excel files from '{input_folder}' and saving to '{output_folder}'...")
    processed_files = process_compound_excels(input_folder, output_folder, z_value)
    print("\nProcessed files:")
    for csv_path in processed_files:
        print(f"  {csv_path}")
        base = os.path.splitext(csv_path)[0]
        excel_path = base + "_analysis.xlsx"
        save_analysis_excel(csv_path, excel_path, z_list, z_input=z_value)

if __name__ == "__main__":
    main()

import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import xlsxwriter

def add_threshold_column(df, z_value):
    df = df.copy()
    if "critical_value" in df.columns:
        df["threshold"] = df["critical_value"] * z_value
    else:
        df["threshold"] = None
    return df

def classify(df):
    concentration_col = "Concentration\n(ng)"
    intensity_col = "Intensity\n(cps)"
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
            for start_col in range(1, n_cols, 6):
                title_cell = df.iloc[2, start_col]
                if isinstance(title_cell, str) and title_cell.startswith("Compound"):
                    compound_title = title_cell.strip()
                    try:
                        cv_val = float(df.iloc[0, start_col + 1])
                    except (ValueError, TypeError, IndexError):
                        cv_val = None
                    headers = df.iloc[3, start_col:start_col+6].tolist()
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

def optimize_z_for_min_false_negatives(sub, z_range=None):
    if z_range is None:
        z_range = np.linspace(0.1, 10, 200)
    best_z = None
    min_fn = float('inf')
    best_stats = None
    concentration_col = "Concentration\n(ng)"
    intensity_col = "Intensity\n(cps)"
    cv = sub['critical_value'].iloc[0]
    for z in z_range:
        threshold = cv * z
        def classify_row(row):
            try:
                concentration = float(row[concentration_col])
                intensity = float(row[intensity_col])
            except (KeyError, ValueError, TypeError):
                return "Unclassified"
            if concentration != 0:
                return "True Positive" if intensity > threshold else "False Negative"
            else:
                return "False Positive" if intensity > threshold else "True Negative"
        classifications = sub.apply(classify_row, axis=1)
        tp = (classifications == "True Positive").sum()
        tn = (classifications == "True Negative").sum()
        fp = (classifications == "False Positive").sum()
        fn = (classifications == "False Negative").sum()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        f1 = (2 * precision * recall) / (precision + recall) if (precision is not None and recall is not None and (precision + recall) > 0) else None
        if fn < min_fn or (fn == min_fn and (best_z is None or z < best_z)):
            min_fn = fn
            best_z = z
            best_stats = (tp, tn, fp, fn, acc, recall, specificity, precision, f1)
    return best_z, min_fn, best_stats

def safe_excel_value(val):
    if pd.isna(val) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return ""
    if isinstance(val, str) and (val.strip() == "" or val.lower() in {"nan", "inf", "-inf"}):
        return ""
    return val

def save_analysis_excel(input_csv, output_excel, z_input=5.5):
    df = pd.read_csv(input_csv)
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        worksheet = writer.sheets['Data']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_len + 2)
        # Sheet 2: Per-compound stats for both original and optimal Z
        compounds = df['compound_id'].unique()
        summary_rows = []
        for compound in compounds:
            sub = df[df['compound_id'] == compound]
            compound_title = sub['compound_title'].iloc[0]
            cv = sub['critical_value'].iloc[0]
            # 1. Stats for original input Z
            threshold_input = cv * z_input
            def classify_at_z(row, z):
                try:
                    concentration = float(row["Concentration\n(ng)"])
                    intensity = float(row["Intensity\n(cps)"])
                except (KeyError, ValueError, TypeError):
                    return "Unclassified"
                threshold = cv * z
                if concentration != 0:
                    return "True Positive" if intensity > threshold else "False Negative"
                else:
                    return "False Positive" if intensity > threshold else "True Negative"
            classifications_input = sub.apply(lambda row: classify_at_z(row, z_input), axis=1)
            tp_i = (classifications_input == "True Positive").sum()
            tn_i = (classifications_input == "True Negative").sum()
            fp_i = (classifications_input == "False Positive").sum()
            fn_i = (classifications_input == "False Negative").sum()
            total_i = tp_i + tn_i + fp_i + fn_i
            acc_i = (tp_i + tn_i) / total_i if total_i > 0 else None
            recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else None
            specificity_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else None
            precision_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else None
            f1_i = (2 * precision_i * recall_i) / (precision_i + recall_i) if (precision_i is not None and recall_i is not None and (precision_i + recall_i) > 0) else None
            # 2. Stats for optimal z
            best_z, min_fn, stats = optimize_z_for_min_false_negatives(sub)
            tp, tn, fp, fn, acc, recall, specificity, precision, f1 = stats
            threshold = cv * best_z
            # Compute ROC/AUC
            y_true = sub["True Label"]
            y_score = sub["Intensity\n(cps)"]
            mask = (~pd.isna(y_score)) & (~pd.isna(y_true))
            y_true = y_true[mask]
            y_score = y_score[mask]
            auc_val = None
            if len(set(y_true)) > 1:
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc_val = auc(fpr, tpr)
            summary_rows.append({
                "Compound": compound_title,
                "Critical Value": cv,
                "Z_Type": "Input Z",
                "Z_Value": z_input,
                "Threshold": threshold_input,
                "TP": tp_i,
                "TN": tn_i,
                "FP": fp_i,
                "FN": fn_i,
                "Accuracy": acc_i,
                "Sensitivity": recall_i,
                "Specificity": specificity_i,
                "Precision": precision_i,
                "F1 Score": f1_i,
                "AUC": auc_val
            })
            summary_rows.append({
                "Compound": compound_title,
                "Critical Value": cv,
                "Z_Type": "Optimal Z (Min FN)",
                "Z_Value": best_z,
                "Threshold": threshold,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": acc,
                "Sensitivity": recall,
                "Specificity": specificity,
                "Precision": precision,
                "F1 Score": f1,
                "AUC": auc_val
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, index=False, sheet_name="Optimal_z_Summary")
        ws2 = writer.sheets["Optimal_z_Summary"]
        for i, col in enumerate(summary_df.columns):
            max_len = max(summary_df[col].astype(str).map(len).max(), len(col))
            ws2.set_column(i, i, max_len + 2)
        # Sheet 3: ROC curves for all compounds (curved scatter with markers)
        roc_ws = writer.book.add_worksheet('ROC_Curves')
        writer.sheets['ROC_Curves'] = roc_ws
        chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        row_cursor = 0
        for compound in compounds:
            sub = df[df['compound_id'] == compound]
            compound_title = sub['compound_title'].iloc[0]
            y_true = sub["True Label"]
            y_score = sub["Intensity\n(cps)"]
            mask = (~pd.isna(y_score)) & (~pd.isna(y_true))
            y_true = y_true[mask]
            y_score = y_score[mask]
            if len(set(y_true)) < 2:
                roc_ws.write(row_cursor, 0, f"{compound_title}: Only one class present, skipping ROC.")
                row_cursor += 2
                continue
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds})
            roc_ws.write(row_cursor, 0, f"{compound_title} ROC (AUC={auc_val:.3f})")
            for j, colname in enumerate(roc_df.columns):
                roc_ws.write(row_cursor+1, j, colname)
            for i, (_, row) in enumerate(roc_df.iterrows()):
                for j, val in enumerate(row):
                    roc_ws.write(row_cursor+2+i, j, safe_excel_value(np.round(val, 6)))
            chart.add_series({
                'name': f'{compound_title} (AUC={auc_val:.3f})',
                'categories': ['ROC_Curves', row_cursor+2, 0, row_cursor+2+len(fpr)-1, 0],
                'values':     ['ROC_Curves', row_cursor+2, 1, row_cursor+2+len(tpr)-1, 1],
                'marker': {'type': 'circle', 'size': 6},
                'line':   {'width': 2},
            })
            row_cursor += 3 + len(roc_df) + 1
        # Add 50% accuracy (chance) diagonal line
        chance_fpr_col = 10
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
        chart.set_title({'name': 'All Compounds ROC Curves (Curved Scatter)'})
        chart.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_legend({'position': 'bottom'})
        roc_ws.insert_chart(row_cursor+2, 0, chart, {'x_scale': 2, 'y_scale': 2})
    print(f"Excel analysis file (optimal z, confusion, ROC, AUC) saved to {output_excel}")

def main():
    input_folder = "input"
    output_folder = "output"
    z_value = 5.5
    print(f"Processing Excel files from '{input_folder}' and saving to '{output_folder}'...")
    processed_files = process_compound_excels(input_folder, output_folder, z_value)
    print("\nProcessed files:")
    for csv_path in processed_files:
        print(f"  {csv_path}")
        base = os.path.splitext(csv_path)[0]
        excel_path = base + "_analysis.xlsx"
        save_analysis_excel(csv_path, excel_path, z_input=z_value)

if __name__ == "__main__":
    main()

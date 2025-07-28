import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import xlsxwriter

def load_individual_crit_values(script_dir, max_compounds=4):
    crit_values = []
    for i in range(1, max_compounds+1):
        filename = os.path.join(script_dir, f'crit_values {i}.txt')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = [float(line.strip()) for line in f if line.strip() and 'crit' not in line.lower()]
            crit_values.append(np.array(lines))
        else:
            crit_values.append(np.array([]))
    return crit_values

def classify(df, crit_value):
    concentration_col = [col for col in df.columns if 'Concentration' in col][0]
    intensity_col = [col for col in df.columns if 'Intensity' in col][0]
    df["threshold"] = crit_value

    def lab(row):
        if row[concentration_col] != 0:
            return "True Positive" if row[intensity_col] > crit_value else "False Negative"
        else:
            return "False Positive" if row[intensity_col] > crit_value else "True Negative"

    df["Classification"] = df.apply(lab, axis=1)
    return df

def process_compound_excels(input_folder, output_folder):
    processed_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(input_folder, filename), header=None)
            sample_id_col = df.iloc[4:, 0].reset_index(drop=True)
            sample_id_header = df.iloc[3, 0]
            file_data = []

            for start_col in range(1, df.shape[1], 6):
                headers = df.iloc[3, start_col:start_col+6].tolist()
                if all(pd.isna(h) for h in headers):
                    continue
                compound_title = df.iloc[2, start_col]
                data_block = df.iloc[4:, start_col:start_col+6]
                data_block.columns = headers
                data_block.insert(0, sample_id_header, sample_id_col)
                data_block = data_block.dropna(subset=[sample_id_header])
                data_block.insert(0, "compound_title", compound_title)
                data_block.insert(0, "compound_id", str(compound_title).replace(" ", "_"))
                file_data.append(data_block)

            output_df = pd.concat(file_data, ignore_index=True)
            out_csv = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_processed.csv")
            output_df.to_csv(out_csv, index=False)
            processed_files.append(out_csv)
    return processed_files

def save_analysis_excel(input_csv, output_excel, crit_values_list):
    df = pd.read_csv(input_csv)
    compounds = df['compound_id'].unique()

    # For the Data tab, do not add classification/threshold columns
    data_for_excel = df.copy()

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        data_for_excel.to_excel(writer, sheet_name='Data', index=False)
        summary_rows = []

        # --- Confusion Matrices Sheet ---
        conf_ws = writer.book.add_worksheet('Confusion_Matrices')
        conf_row = 0
        for idx, compound in enumerate(compounds):
            sub = df[df['compound_id'] == compound]
            y_true = (sub[[c for c in sub.columns if 'Concentration' in c][0]] != 0).astype(int)
            y_score = sub[[c for c in sub.columns if 'Intensity' in c][0]]
            crit_values = crit_values_list[idx] if idx < len(crit_values_list) else np.array([])

            conf_ws.write(conf_row, 0, f"Compound: {compound}")
            conf_row += 1
            conf_ws.write_row(conf_row, 1, ['Threshold', 'TP', 'FN', 'FP', 'TN'])
            conf_row += 1

            for crit in crit_values:
                classified = classify(sub.copy(), crit)
                tp = len(classified[classified["Classification"] == "True Positive"])
                fn = len(classified[classified["Classification"] == "False Negative"])
                fp = len(classified[classified["Classification"] == "False Positive"])
                tn = len(classified[classified["Classification"] == "True Negative"])
                conf_ws.write_row(conf_row, 1, [crit, tp, fn, fp, tn])
                conf_row += 1

            conf_row += 2  # Space between compounds

        roc_ws = writer.book.add_worksheet('ROC_Curves')
        chart = writer.book.add_chart({'type': 'scatter'})
        chart_smooth = writer.book.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        row_cursor = 0

        for idx, compound in enumerate(compounds):
            sub = df[df['compound_id'] == compound]
            y_true = (sub[[c for c in sub.columns if 'Concentration' in c][0]] != 0).astype(int)
            y_score = sub[[c for c in sub.columns if 'Intensity' in c][0]]

            crit_values = crit_values_list[idx] if idx < len(crit_values_list) else np.array([])

            # --- Non-smoothed: Only crit_values for this compound ---
            sweep_crits_nonsmooth = np.sort(np.unique(crit_values))
            fpr_ns, tpr_ns, crits_ns = [], [], []
            for crit in sweep_crits_nonsmooth:
                classified = classify(sub.copy(), crit)
                tp = len(classified[classified["Classification"] == "True Positive"])
                tn = len(classified[classified["Classification"] == "True Negative"])
                fp = len(classified[classified["Classification"] == "False Positive"])
                fn = len(classified[classified["Classification"] == "False Negative"])
                if (fp + tn) == 0 or (tp + fn) == 0:
                    continue
                fpr_ns.append(fp / (fp + tn))
                tpr_ns.append(tp / (tp + fn))
                crits_ns.append(crit)
            auc_ns = auc(fpr_ns, tpr_ns) if len(fpr_ns) > 1 else float('nan')

            # --- Smoothed: Crit values + all unique intensities for this compound ---
            sweep_crits_smooth = np.sort(np.unique(np.concatenate([y_score.values, crit_values])))
            fpr_sm, tpr_sm, crits_sm = [], [], []
            for crit in sweep_crits_smooth:
                classified = classify(sub.copy(), crit)
                tp = len(classified[classified["Classification"] == "True Positive"])
                tn = len(classified[classified["Classification"] == "True Negative"])
                fp = len(classified[classified["Classification"] == "False Positive"])
                fn = len(classified[classified["Classification"] == "False Negative"])
                if (fp + tn) == 0 or (tp + fn) == 0:
                    continue
                fpr_sm.append(fp / (fp + tn))
                tpr_sm.append(tp / (tp + fn))
                crits_sm.append(crit)
            auc_sm = auc(fpr_sm, tpr_sm) if len(fpr_sm) > 1 else float('nan')

            # --- Write Non-smoothed table ---
            roc_ws.write(row_cursor, 0, f"{compound} ROC (AUC Crits Only={auc_ns:.3f})")
            roc_ws.write_row(row_cursor+1, 0, ['FPR', 'TPR', 'Crit_Value'])
            for i, data in enumerate(zip(fpr_ns, tpr_ns, crits_ns)):
                roc_ws.write_row(row_cursor+2+i, 0, data)
            chart.add_series({
                'name': f'{compound} Crits Only',
                'categories': ['ROC_Curves', row_cursor+2, 0, row_cursor+1+len(fpr_ns), 0],
                'values': ['ROC_Curves', row_cursor+2, 1, row_cursor+1+len(tpr_ns), 1],
                'marker': {'type': 'diamond', 'size': 7},
                'line': {'none': True},
            })

            row_cursor_smooth = row_cursor + 2 + len(fpr_ns) + 1
            # --- Write Smoothed table ---
            roc_ws.write(row_cursor_smooth, 0, f"{compound} ROC (AUC Crits+All Unique={auc_sm:.3f})")
            roc_ws.write_row(row_cursor_smooth+1, 0, ['FPR', 'TPR', 'Crit_Value'])
            for i, data in enumerate(zip(fpr_sm, tpr_sm, crits_sm)):
                roc_ws.write_row(row_cursor_smooth+2+i, 0, data)
            chart_smooth.add_series({
                'name': f'{compound} Crits+All Unique',
                'categories': ['ROC_Curves', row_cursor_smooth+2, 0, row_cursor_smooth+1+len(fpr_sm), 0],
                'values': ['ROC_Curves', row_cursor_smooth+2, 1, row_cursor_smooth+1+len(tpr_sm), 1],
                'marker': {'type': 'circle', 'size': 5},
                'line': {'width': 2},
            })

            # Move cursor for next compound
            row_cursor = row_cursor_smooth + 2 + len(fpr_sm) + 2

            summary_rows.append({"Compound": compound, "AUC_CritsOnly": auc_ns, "AUC_CritsAllUnique": auc_sm})

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="AUC_Summary", index=False)

        # Add 50% chance line to both charts
        for c in [chart, chart_smooth]:
            c.add_series({
                'name': 'Chance',
                'categories': ['ROC_Curves', 1, 0, 2, 0],
                'values':     ['ROC_Curves', 1, 1, 2, 1],
                'marker': {'type': 'none'},
                'line': {'color': 'gray', 'dash_type': 'dash', 'width': 1.5},
            })
            c.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
            c.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
            c.set_legend({'position': 'bottom'})

        chart.set_title({'name': 'ROC Curve Using Given Threshold Values'})
        chart_smooth.set_title({'name': 'ROC Curve Using All Possible Thresholds'})
        roc_ws.insert_chart('M2', chart, {'x_scale': 1.3, 'y_scale': 1.3})
        roc_ws.insert_chart('M22', chart_smooth, {'x_scale': 1.3, 'y_scale': 1.3})

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input")
    output_folder = os.path.join(script_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    crit_values_list = load_individual_crit_values(script_dir)

    processed_files = process_compound_excels(input_folder, output_folder)
    for csv_path in processed_files:
        excel_path = csv_path.replace('.csv', '_analysis.xlsx')
        save_analysis_excel(csv_path, excel_path, crit_values_list)
        print(f"Analysis complete for: {os.path.basename(csv_path)}")

if __name__ == "__main__":
    main()

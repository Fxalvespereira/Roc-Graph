import os
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import xlsxwriter

def load_crit_values_from_txt(script_dir):
    filepath = os.path.join(script_dir, 'crit_values.txt')
    if not os.path.exists(filepath):
        print("crit_values.txt not found, only using unique intensities.")
        return np.array([])
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return np.array([float(line.strip()) for line in lines if line.strip() and not line.lower().startswith('crit')])

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

def save_analysis_excel(input_csv, output_excel, crit_values):
    df = pd.read_csv(input_csv)
    compounds = df['compound_id'].unique()

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        summary_rows = []

        roc_ws = writer.book.add_worksheet('ROC_Curves')
        chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'smooth'})
        row_cursor = 0

        for compound in compounds:
            sub = df[df['compound_id'] == compound]
            y_true = (sub[[c for c in sub.columns if 'Concentration' in c][0]] != 0).astype(int)
            y_score = sub[[c for c in sub.columns if 'Intensity' in c][0]]

            # Use all unique observed intensities AND crit values from txt
            sweep_crits = np.sort(np.unique(np.concatenate([y_score.values, crit_values])))

            fpr, tpr, crits = [], [], []
            best_acc, best_crit = -1, None
            for crit in sweep_crits:
                classified = classify(sub.copy(), crit)
                tp = len(classified[classified["Classification"] == "True Positive"])
                tn = len(classified[classified["Classification"] == "True Negative"])
                fp = len(classified[classified["Classification"] == "False Positive"])
                fn = len(classified[classified["Classification"] == "False Negative"])
                total = tp + tn + fp + fn
                acc = (tp + tn) / total if total > 0 else 0
                if acc > best_acc:
                    best_acc, best_crit = acc, crit

                if (fp + tn) == 0 or (tp + fn) == 0:
                    continue
                fpr.append(fp / (fp + tn))
                tpr.append(tp / (tp + fn))
                crits.append(crit)

            if len(fpr) < 2:
                continue

            sorted_pairs = sorted(zip(fpr, tpr, crits))
            fpr_sorted, tpr_sorted, crits_sorted = zip(*sorted_pairs)
            auc_val = auc(fpr_sorted, tpr_sorted)

            roc_ws.write(row_cursor, 0, f"{compound} ROC (AUC={auc_val:.3f})")
            roc_ws.write_row(row_cursor+1, 0, ['FPR', 'TPR', 'Crit_Value'])
            for idx, data in enumerate(zip(fpr_sorted, tpr_sorted, crits_sorted)):
                roc_ws.write_row(row_cursor+2+idx, 0, data)

            chart.add_series({
                'name': compound,
                'categories': ['ROC_Curves', row_cursor+2, 0, row_cursor+1+len(fpr_sorted), 0],
                'values': ['ROC_Curves', row_cursor+2, 1, row_cursor+1+len(tpr_sorted), 1],
                'marker': {'type': 'circle', 'size': 5},
                'line': {'width': 2},
            })

            row_cursor += len(fpr_sorted) + 4
            summary_rows.append({"Compound": compound, "Best_Crit": best_crit, "Accuracy": best_acc})

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Optimal_Crit_Summary", index=False)

        # Add 50% accuracy (chance) diagonal line
        roc_ws.write(row_cursor, 0, "FPR_Chance")
        roc_ws.write(row_cursor, 1, "TPR_Chance")
        roc_ws.write(row_cursor+1, 0, 0)
        roc_ws.write(row_cursor+1, 1, 0)
        roc_ws.write(row_cursor+2, 0, 1)
        roc_ws.write(row_cursor+2, 1, 1)
        chart.add_series({
            'name': '50% Accuracy (Chance)',
            'categories': ['ROC_Curves', row_cursor+1, 0, row_cursor+2, 0],
            'values':     ['ROC_Curves', row_cursor+1, 1, row_cursor+2, 1],
            'marker': {'type': 'none'},
            'line': {'color': 'gray', 'dash_type': 'dash', 'width': 1.5},
        })

        chart.set_title({'name': 'ROC Curves per Compound'})
        chart.set_x_axis({'name': 'False Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1, 'major_gridlines': {'visible': True}})
        chart.set_legend({'position': 'bottom'})

        roc_ws.insert_chart('E2', chart, {'x_scale': 2, 'y_scale': 2})

def plot_and_save_roc_png(input_csv, output_png, crit_values):
    df = pd.read_csv(input_csv)
    compounds = df['compound_id'].unique()
    plt.figure(figsize=(7, 7))
    found_data = False
    for compound in compounds:
        sub = df[df['compound_id'] == compound]
        y_true = (sub[[c for c in sub.columns if 'Concentration' in c][0]] != 0).astype(int)
        y_score = sub[[c for c in sub.columns if 'Intensity' in c][0]]

        # Use all unique observed intensities AND crit values from txt
        sweep_crits = np.sort(np.unique(np.concatenate([y_score.values, crit_values])))

        tpr = []
        fpr = []
        for crit in sweep_crits:
            preds = (y_score > crit).astype(int)
            tn = ((preds == 0) & (y_true == 0)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            tp = ((preds == 1) & (y_true == 1)).sum()
            if (fp + tn) == 0 or (tp + fn) == 0:
                continue
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))
        if len(fpr) < 2:
            continue
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{compound} (AUC = {roc_auc:.3f})')
        found_data = True
    if not found_data:
        print(f"No valid ROC curve data for: {os.path.basename(input_csv)}")
        plt.close()
        return
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve(s) per Compound')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"ROC curve PNG saved to: {output_png}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input")
    output_folder = os.path.join(script_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    crit_values = load_crit_values_from_txt(script_dir)

    processed_files = process_compound_excels(input_folder, output_folder)
    for csv_path in processed_files:
        excel_path = csv_path.replace('.csv', '_crit_analysis.xlsx')
        save_analysis_excel(csv_path, excel_path, crit_values)
        print(f"Analysis complete for: {os.path.basename(csv_path)}")
        output_png = excel_path.replace('_crit_analysis.xlsx', '_ROC.png')
        plot_and_save_roc_png(csv_path, output_png, crit_values)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import xlsxwriter

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def load_and_clean_data(filepath):
    Compound_G = pd.read_excel(
        filepath,
        skiprows=[0, 4],
        nrows=52,
        usecols="A:G",
        names=[
            "Sample ID", "Concentration\n(ng)", "Intensity\n(cps)", "Mass\n(m/z)",
            "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"
        ]
    )
    Compound_F = pd.read_excel(
        filepath,
        skiprows=[0, 1],
        nrows=52,
        usecols="A,H:M",
        names=[
            "Sample ID", "Concentration\n(ng)", "Intensity\n(cps)", "Mass\n(m/z)",
            "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"
        ]
    )
    Compound_G = Compound_G.dropna().iloc[1:].reset_index(drop=True)
    Compound_F = Compound_F.dropna().iloc[1:].reset_index(drop=True)
    return Compound_G, Compound_F

def extract_cv_lod(filepath):
    sheet = pd.read_excel(filepath, sheet_name="Sheet1", header=None)
    lod90_g = sheet.iloc[0, 2]
    cv90_g = sheet.iloc[1, 2]
    lod90_f = sheet.iloc[0, 8]
    cv90_f = sheet.iloc[1, 8]
    return {
        "Compound_G": {"Cv90": cv90_g, "LOD90": lod90_g},
        "Compound_F": {"Cv90": cv90_f, "LOD90": lod90_f}
    }

def classify_samples(df, cv90, lod90):
    threshold = cv90 * lod90
    blanks_df = df[df["Concentration\n(ng)"] == 0].reset_index(drop=True)
    non_blanks_df = df[df["Concentration\n(ng)"] != 0].reset_index(drop=True)

    blank_results = blanks_df[["Sample ID", "Intensity\n(cps)"]].copy()
    blank_results["Above Threshold"] = blanks_df["Intensity\n(cps)"] > threshold

    non_blank_results = non_blanks_df[["Sample ID", "Intensity\n(cps)"]].copy()
    non_blank_results["Above Threshold"] = non_blanks_df["Intensity\n(cps)"] > threshold

    return blanks_df, non_blanks_df, blank_results, non_blank_results

# Use previously defined interactive chart function here
from pathlib import Path

def generate_roc_auc_excel_interactive(
    compound_g_blanks, compound_g_non_blanks,
    compound_f_blanks, compound_f_non_blanks,
    result_blanks_g, result_non_blanks_g,
    result_blanks_f, result_non_blanks_f,
    output_path):

    def compute_roc_auc(blank_df, non_blank_df):
        y_true = [0] * len(blank_df) + [1] * len(non_blank_df)
        y_scores = list(blank_df["Intensity\n(cps)"]) + list(non_blank_df["Intensity\n(cps)"])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def compute_non_blank_auc(non_blank_df):
        scores = list(non_blank_df["Intensity\n(cps)"])
        threshold = pd.Series(scores).quantile(0.2)
        y_true = [0 if val < threshold else 1 for val in scores]
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    # Compute both types
    fpr_g_all, tpr_g_all, auc_g_all = compute_roc_auc(compound_g_blanks, compound_g_non_blanks)
    fpr_f_all, tpr_f_all, auc_f_all = compute_roc_auc(compound_f_blanks, compound_f_non_blanks)

    fpr_g_nb, tpr_g_nb, auc_g_nb = compute_non_blank_auc(compound_g_non_blanks)
    fpr_f_nb, tpr_f_nb, auc_f_nb = compute_non_blank_auc(compound_f_non_blanks)

    auc_summary = pd.DataFrame({
        "Compound": ["Compound G", "Compound F"],
        "AUC (With Blanks)": [auc_g_all, auc_f_all],
        "AUC (Non-Blanks Only)": [auc_g_nb, auc_f_nb]
    })

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Output result summaries
        auc_summary.to_excel(writer, sheet_name='Summary', startrow=0, startcol=0, index=False)
        result_blanks_g.to_excel(writer, sheet_name='Summary', startrow=4, startcol=0, index=False)
        result_non_blanks_g.to_excel(writer, sheet_name='Summary', startrow=4, startcol=3, index=False)
        result_blanks_f.to_excel(writer, sheet_name='Summary', startrow=4, startcol=6, index=False)
        result_non_blanks_f.to_excel(writer, sheet_name='Summary', startrow=4, startcol=9, index=False)

        # Add ROC data for charting
        roc_sheet = workbook.add_worksheet("ROC_Data")
        writer.sheets["ROC_Data"] = roc_sheet
        headers = [
            ("FPR_G_All", fpr_g_all), ("TPR_G_All", tpr_g_all),
            ("FPR_F_All", fpr_f_all), ("TPR_F_All", tpr_f_all),
            ("FPR_G_NB", fpr_g_nb), ("TPR_G_NB", tpr_g_nb),
            ("FPR_F_NB", fpr_f_nb), ("TPR_F_NB", tpr_f_nb),
        ]
        for i, (label, data) in enumerate(headers):
            roc_sheet.write(0, i, label)
            for j, val in enumerate(data):
                roc_sheet.write(j+1, i, val)

        def add_chart(sheet_name, col_pairs, title, pos):
            chart = workbook.add_chart({'type': 'line'})
            chart.set_title({'name': title})
            chart.set_x_axis({'name': 'False Positive Rate'})
            chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1})
            chart.set_legend({'position': 'bottom'})
            colors = ['blue', 'red']
            labels = ['Compound G', 'Compound F']
            for i, (x_col, y_col) in enumerate(col_pairs):
                chart.add_series({
                    'name': labels[i],
                    'categories': ['ROC_Data', 1, x_col, len(fpr_g_all), x_col],
                    'values':     ['ROC_Data', 1, y_col, len(fpr_g_all), y_col],
                    'line':       {'color': colors[i]},
                })
            writer.sheets[sheet_name].insert_chart(pos, chart)

        add_chart('Summary', [(0, 1), (2, 3)], "ROC (with blanks)", 'M2')
        add_chart('Summary', [(4, 5), (6, 7)], "ROC (non-blanks only)", 'M20')

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_file = os.path.splitext(filename)[0] + "_results.xlsx"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)

            print(f"🔄 Processing {filename}...")

            try:
                Compound_G, Compound_F = load_and_clean_data(input_path)
                results = extract_cv_lod(input_path)

                blanks_g, non_blanks_g, result_blanks_g, result_non_blanks_g = classify_samples(
                    Compound_G, results["Compound_G"]["Cv90"], results["Compound_G"]["LOD90"]
                )
                blanks_f, non_blanks_f, result_blanks_f, result_non_blanks_f = classify_samples(
                    Compound_F, results["Compound_F"]["Cv90"], results["Compound_F"]["LOD90"]
                )

                generate_roc_auc_excel_interactive(
                    blanks_g, non_blanks_g, blanks_f, non_blanks_f,
                    result_blanks_g, result_non_blanks_g,
                    result_blanks_f, result_non_blanks_f,
                    output_path
                )

                print(f"✅ Saved to: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
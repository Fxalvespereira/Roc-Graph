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

def classify_samples_whole(df, cv90, z):
    threshold = cv90 * z
    results = df[["Sample ID", "Concentration\n(ng)", "Intensity\n(cps)"]].copy()
    results["Above Threshold"] = df["Intensity\n(cps)"] > threshold
    results["True Label"] = (df["Concentration\n(ng)"] != 0).astype(int)
    return results

def generate_roc_auc_excel_interactive_whole(
    results_g, results_f, output_path):

    # ROC calculation for Compound G
    y_true_g = results_g["True Label"]
    y_score_g = results_g["Intensity\n(cps)"]
    fpr_g, tpr_g, _ = roc_curve(y_true_g, y_score_g)
    auc_g = auc(fpr_g, tpr_g)

    # ROC calculation for Compound F
    y_true_f = results_f["True Label"]
    y_score_f = results_f["Intensity\n(cps)"]
    fpr_f, tpr_f, _ = roc_curve(y_true_f, y_score_f)
    auc_f = auc(fpr_f, tpr_f)

    auc_summary = pd.DataFrame({
        "Compound": ["Compound G", "Compound F"],
        "AUC": [auc_g, auc_f]
    })

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        auc_summary.to_excel(writer, sheet_name='Summary', startrow=0, startcol=0, index=False)
        results_g.to_excel(writer, sheet_name='Summary', startrow=4, startcol=0, index=False)
        results_f.to_excel(writer, sheet_name='Summary', startrow=4, startcol=6, index=False)

        # Add ROC data for charting
        roc_sheet = workbook.add_worksheet("ROC_Data")
        writer.sheets["ROC_Data"] = roc_sheet
        # Write Compound G
        roc_sheet.write(0, 0, "FPR_G")
        roc_sheet.write(0, 1, "TPR_G")
        for i, (fpr, tpr) in enumerate(zip(fpr_g, tpr_g)):
            roc_sheet.write(i+1, 0, fpr)
            roc_sheet.write(i+1, 1, tpr)
        # Write Compound F
        roc_sheet.write(0, 3, "FPR_F")
        roc_sheet.write(0, 4, "TPR_F")
        for i, (fpr, tpr) in enumerate(zip(fpr_f, tpr_f)):
            roc_sheet.write(i+1, 3, fpr)
            roc_sheet.write(i+1, 4, tpr)

        # Add interactive chart
        chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
        chart.set_title({'name': 'ROC Curve (Scatter)'})
        chart.set_x_axis({'name': 'False Positive Rate'})
        chart.set_y_axis({'name': 'True Positive Rate', 'min': 0, 'max': 1})
        chart.set_legend({'position': 'bottom'})

        # Compound G
        chart.add_series({
            'name':       'Compound G',
            'categories': ['ROC_Data', 1, 0, len(fpr_g), 0],  # X = FPR_G
            'values':     ['ROC_Data', 1, 1, len(tpr_g), 1],  # Y = TPR_G
            'marker':     {'type': 'circle', 'size': 5, 'border': {'color': 'blue'}, 'fill': {'color': 'blue'}},
            'line':       {'color': 'blue'}
        })

        # Compound F
        chart.add_series({
            'name':       'Compound F',
            'categories': ['ROC_Data', 1, 3, len(fpr_f), 3],  # X = FPR_F
            'values':     ['ROC_Data', 1, 4, len(tpr_f), 4],  # Y = TPR_F
            'marker':     {'type': 'square', 'size': 5, 'border': {'color': 'red'}, 'fill': {'color': 'red'}},
            'line':       {'color': 'red'}
        })

        # Insert the chart into the "Summary" sheet
        writer.sheets['Summary'].insert_chart('M2', chart)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    z = 1.1
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_file = os.path.splitext(filename)[0] + "_results.xlsx"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)

            print(f"🔄 Processing {filename}...")

            try:
                Compound_G, Compound_F = load_and_clean_data(input_path)
                results = extract_cv_lod(input_path)

                results_g = classify_samples_whole(
                    Compound_G, results["Compound_G"]["Cv90"], z
                )
                results_f = classify_samples_whole(
                    Compound_F, results["Compound_F"]["Cv90"], z
                )

                generate_roc_auc_excel_interactive_whole(
                    results_g, results_f, output_path
                )

                print(f"✅ Saved to: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"


def load_and_clean_data(filepath):
    Compound_G = pd.read_excel(
        filepath,
        skiprows=[0, 4],
        nrows=52,
        usecols="A:C",
        names=["Sample ID", "Concentration (ng)", "Intensity\n(cps)"]
    ).dropna().iloc[1:].reset_index(drop=True)

    Compound_F = pd.read_excel(
        filepath,
        skiprows=[0, 1],
        nrows=52,
        usecols="A,H:I",
        names=["Sample ID", "Concentration (ng)", "Intensity\n(cps)"]
    ).dropna().iloc[1:].reset_index(drop=True)

    return Compound_G, Compound_F


def extract_cv_values(filepath):
    sheet = pd.read_excel(filepath, sheet_name=0, header=None)
    cv_g = None
    cv_f = None

    for row in range(sheet.shape[0]):
        for col in range(sheet.shape[1]):
            cell = str(sheet.iat[row, col]).strip().lower()
            if "cv90" in cell:
                if col == 1:
                    cv_g = pd.to_numeric(sheet.iat[row, col + 1], errors='coerce')
                elif col == 7:
                    cv_f = pd.to_numeric(sheet.iat[row, col + 1], errors='coerce')

    if cv_g is None or cv_f is None:
        raise ValueError("Cv90 value not found for both compounds. Please check the Excel format.")

    return {
        "Compound_G": cv_g,
        "Compound_F": cv_f
    }


def classify_samples(df, threshold):
    conditions = [
        (df["Concentration (ng)"] != 0) & (df["Intensity\n(cps)"] > threshold),
        (df["Concentration (ng)"] != 0) & (df["Intensity\n(cps)"] <= threshold),
        (df["Concentration (ng)"] == 0) & (df["Intensity\n(cps)"] > threshold),
        (df["Concentration (ng)"] == 0) & (df["Intensity\n(cps)"] <= threshold)
    ]
    choices = ['TP', 'FN', 'FP', 'TN']
    df['Classification'] = np.select(conditions, choices, default='Unknown')
    df['Above Threshold'] = df['Intensity\n(cps)'] > threshold
    df['True Label'] = (df['Concentration (ng)'] != 0).astype(int)

    return df


def add_interactive_roc_chart(workbook, worksheet, fpr_g, tpr_g, fpr_f, tpr_f):
    roc_sheet = workbook.add_worksheet("ROC_Data")
    roc_sheet.write(0, 0, "FPR_G")
    roc_sheet.write(0, 1, "TPR_G")
    roc_sheet.write(0, 3, "FPR_F")
    roc_sheet.write(0, 4, "TPR_F")

    for i, (fg, tg) in enumerate(zip(fpr_g, tpr_g)):
        roc_sheet.write(i + 1, 0, fg)
        roc_sheet.write(i + 1, 1, tg)
    for i, (ff, tf) in enumerate(zip(fpr_f, tpr_f)):
        roc_sheet.write(i + 1, 3, ff)
        roc_sheet.write(i + 1, 4, tf)

    chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
    chart.add_series({
        'name': 'Compound G',
        'categories': ['ROC_Data', 1, 0, len(fpr_g), 0],
        'values': ['ROC_Data', 1, 1, len(tpr_g), 1],
        'marker': {'type': 'circle'},
        'line': {'color': 'blue'},
    })
    chart.add_series({
        'name': 'Compound F',
        'categories': ['ROC_Data', 1, 3, len(fpr_f), 3],
        'values': ['ROC_Data', 1, 4, len(tpr_f), 4],
        'marker': {'type': 'square'},
        'line': {'color': 'red'},
    })
    chart.set_title({'name': 'ROC Curve'})
    chart.set_x_axis({'name': 'False Positive Rate'})
    chart.set_y_axis({'name': 'True Positive Rate'})
    chart.set_legend({'position': 'bottom'})

    worksheet.insert_chart('G2', chart)


def process_file(filepath, z=1.1):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{basename}_results.xlsx")

    Compound_G, Compound_F = load_and_clean_data(filepath)
    cv_values = extract_cv_values(filepath)

    threshold_g = cv_values["Compound_G"] * z
    threshold_f = cv_values["Compound_F"] * z

    Compound_G = classify_samples(Compound_G, threshold_g)
    Compound_F = classify_samples(Compound_F, threshold_f)

    y_true_g, y_score_g = Compound_G['True Label'], Compound_G['Intensity\n(cps)']
    y_true_f, y_score_f = Compound_F['True Label'], Compound_F['Intensity\n(cps)']

    fpr_g, tpr_g, thresholds_g = roc_curve(y_true_g, y_score_g)
    auc_g = auc(fpr_g, tpr_g)

    fpr_f, tpr_f, thresholds_f = roc_curve(y_true_f, y_score_f)
    auc_f = auc(fpr_f, tpr_f)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        summary_df = pd.DataFrame({
            "Compound": ["Compound G", "Compound F"],
            "Z-Value": [z, z],
            "Threshold": [threshold_g, threshold_f],
            "AUC": [auc_g, auc_f],
            "Min ROC Threshold": [thresholds_g.min(), thresholds_f.min()],
            "Max ROC Threshold": [thresholds_g.max(), thresholds_f.max()]
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        worksheet = writer.sheets['Summary']

        add_interactive_roc_chart(workbook, worksheet, fpr_g, tpr_g, fpr_f, tpr_f)

        columns_to_export = ["Sample ID", "Concentration (ng)", "Intensity\n(cps)", "Above Threshold", "Classification"]
        Compound_G[columns_to_export].to_excel(writer, sheet_name="Compound_G", index=False)
        Compound_F[columns_to_export].to_excel(writer, sheet_name="Compound_F", index=False)

    print(f"✅ Results written to {output_path}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".xlsx") and not file.startswith('~$'):
            print(f"Processing {file}...")
            process_file(os.path.join(INPUT_FOLDER, file), z=1.1)


if __name__ == "__main__":
    main()

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
        usecols="A:G",
        names=[
            "Sample ID", "Concentration (ng)", "Intensity\n(cps)", "Mass\n(m/z)",
            "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"
        ]
    ).dropna().iloc[1:].reset_index(drop=True)

    Compound_F = pd.read_excel(
        filepath,
        skiprows=[0, 1],
        nrows=52,
        usecols="A,H:M",
        names=[
            "Sample ID", "Concentration (ng)", "Intensity\n(cps)", "Mass\n(m/z)",
            "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"
        ]
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
                if col == 1:  # Column B (index 1) for Compound G
                    cv_g = pd.to_numeric(sheet.iat[row, col + 1], errors='coerce')
                elif col == 7:  # Column H (index 7) for Compound F
                    cv_f = pd.to_numeric(sheet.iat[row, col + 1], errors='coerce')

    if cv_g is None or cv_f is None:
        raise ValueError("Cv90 value not found for both compounds. Please check the Excel format.")

    return {
        "Compound_G": cv_g,
        "Compound_F": cv_f
    }


def classify_samples(df, threshold):
    conditions = [
        (df["Concentration (ng)"] != 0) & (df["Intensity\n(cps)"] > threshold),  # TP
        (df["Concentration (ng)"] != 0) & (df["Intensity\n(cps)"] <= threshold), # FN
        (df["Concentration (ng)"] == 0) & (df["Intensity\n(cps)"] > threshold),  # FP
        (df["Concentration (ng)"] == 0) & (df["Intensity\n(cps)"] <= threshold)  # TN
    ]
    choices = ['TP', 'FN', 'FP', 'TN']
    df['Classification'] = np.select(conditions, choices, default='Unknown')
    df['Above Threshold'] = df['Intensity\n(cps)'] > threshold
    df['True Label'] = (df['Concentration (ng)'] != 0).astype(int)

    return df


def save_combined_roc_curve(fpr_g, tpr_g, auc_g, fpr_f, tpr_f, auc_f, name, output_dir):
    plt.figure()
    plt.plot(fpr_g, tpr_g, color='blue', lw=2, label=f'Compound G (AUC = {auc_g:.2f})')
    plt.plot(fpr_f, tpr_f, color='red', lw=2, label=f'Compound F (AUC = {auc_f:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{name}_combined_roc_curve.png")
    plt.savefig(img_path)
    plt.close()
    return img_path


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

    roc_img_path = save_combined_roc_curve(fpr_g, tpr_g, auc_g, fpr_f, tpr_f, auc_f, basename, OUTPUT_FOLDER)

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
        worksheet.insert_image("G2", roc_img_path)

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

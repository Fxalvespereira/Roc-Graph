import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import xlsxwriter

# === Folder paths ===
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

# === Core Functions ===

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

def generate_roc_auc_excel(compound_g_blanks, compound_g_non_blanks,
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
        # Assume everything is positive, create fake negatives by shuffling
        y_true = [1] * len(non_blank_df)
        y_scores = list(non_blank_df["Intensity\n(cps)"])
        # Fake negatives: take bottom 20% as "negatives" for curve shape
        threshold = pd.Series(y_scores).quantile(0.2)
        y_true = [0 if val < threshold else 1 for val in y_scores]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    # ROC AUC with blanks
    fpr_g_all, tpr_g_all, auc_g_all = compute_roc_auc(compound_g_blanks, compound_g_non_blanks)
    fpr_f_all, tpr_f_all, auc_f_all = compute_roc_auc(compound_f_blanks, compound_f_non_blanks)

    # ROC AUC without blanks (non-blanks only, estimated)
    fpr_g_nb, tpr_g_nb, auc_g_nb = compute_non_blank_auc(compound_g_non_blanks)
    fpr_f_nb, tpr_f_nb, auc_f_nb = compute_non_blank_auc(compound_f_non_blanks)

    # Plot 1: with blanks
    plt.figure()
    plt.plot(fpr_g_all, tpr_g_all, color='blue', lw=2, label=f'G (AUC = {auc_g_all:.2f})')
    plt.plot(fpr_f_all, tpr_f_all, color='red', lw=2, label=f'F (AUC = {auc_f_all:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curve (with blanks)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_with_blanks_path = output_path.replace('.xlsx', '_roc_with_blanks.png')
    plt.savefig(roc_with_blanks_path)
    plt.close()

    # Plot 2: without blanks
    plt.figure()
    plt.plot(fpr_g_nb, tpr_g_nb, color='blue', lw=2, label=f'G (AUC = {auc_g_nb:.2f})')
    plt.plot(fpr_f_nb, tpr_f_nb, color='red', lw=2, label=f'F (AUC = {auc_f_nb:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("ROC Curve (non-blanks only)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    roc_non_blanks_path = output_path.replace('.xlsx', '_roc_non_blanks.png')
    plt.savefig(roc_non_blanks_path)
    plt.close()

    # Summary
    auc_summary = pd.DataFrame({
        "Compound": ["Compound G", "Compound F"],
        "AUC (With Blanks)": [auc_g_all, auc_f_all],
        "AUC (Non-Blanks Only)": [auc_g_nb, auc_f_nb]
    })

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        auc_summary.to_excel(writer, sheet_name='Summary', startrow=0, startcol=0, index=False)
        result_blanks_g.to_excel(writer, sheet_name='Summary', startrow=4, startcol=0, index=False)
        result_non_blanks_g.to_excel(writer, sheet_name='Summary', startrow=4, startcol=3, index=False)
        result_blanks_f.to_excel(writer, sheet_name='Summary', startrow=4, startcol=6, index=False)
        result_non_blanks_f.to_excel(writer, sheet_name='Summary', startrow=4, startcol=9, index=False)

        worksheet = writer.sheets['Summary']
        worksheet.insert_image('M2', roc_with_blanks_path)
        worksheet.insert_image('M20', roc_non_blanks_path)


# === Main Runner ===

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Loop through every Excel file in the input folder
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):  # Skip temp files
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_name = os.path.splitext(filename)[0] + "_results.xlsx"
            output_path = os.path.join(OUTPUT_FOLDER, output_name)

            print(f" Processing {filename}...")

            try:
                # Load data
                Compound_G, Compound_F = load_and_clean_data(input_path)
                results = extract_cv_lod(input_path)

                # Classify samples
                blanks_g, non_blanks_g, result_blanks_g, result_non_blanks_g = classify_samples(
                    Compound_G, results["Compound_G"]["Cv90"], results["Compound_G"]["LOD90"]
                )
                blanks_f, non_blanks_f, result_blanks_f, result_non_blanks_f = classify_samples(
                    Compound_F, results["Compound_F"]["Cv90"], results["Compound_F"]["LOD90"]
                )

                # Generate report
                generate_roc_auc_excel(
                    blanks_g, non_blanks_g, blanks_f, non_blanks_f,
                    result_blanks_g, result_non_blanks_g,
                    result_blanks_f, result_non_blanks_f,
                    output_path
                )

                print(f"Saved: {output_name}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()
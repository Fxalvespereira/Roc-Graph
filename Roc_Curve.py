import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def load_and_clean_data(filepath):
    Compound_G = pd.read_excel(
        filepath, skiprows=[0, 4], nrows=52, usecols="A:G",
        names=["Sample ID", "Concentration\n(ng)", "Intensity\n(cps)", "Mass\n(m/z)",
               "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"]
    )
    Compound_F = pd.read_excel(
        filepath, skiprows=[0, 1], nrows=52, usecols="A,H:M",
        names=["Sample ID", "Concentration\n(ng)", "Intensity\n(cps)", "Mass\n(m/z)",
               "Mass 15 Intensity\n(cps)", "Mass 18 Intensity\n(cps)", "Mass 18 Ratio"]
    )
    Compound_G = Compound_G.dropna().iloc[1:].reset_index(drop=True)
    Compound_F = Compound_F.dropna().iloc[1:].reset_index(drop=True)
    return Compound_G, Compound_F

def extract_cv(filepath):
    sheet = pd.read_excel(filepath, sheet_name="Sheet1", header=None)
    return {
        "Compound_G": {"Cv90": sheet.iloc[1, 2]},
        "Compound_F": {"Cv90": sheet.iloc[1, 8]}
    }

def classify_samples_whole(df, cv90, z):
    threshold = cv90 * z
    results = df[["Sample ID", "Concentration\n(ng)", "Intensity\n(cps)"]].copy()
    results.insert(1, "Concentration (ng)", results.pop("Concentration\n(ng)"))  # Reorder
    results["Above Threshold"] = df["Intensity\n(cps)"] > threshold
    results["True Label"] = (results["Concentration (ng)"] != 0).astype(int)
    return results, threshold

def generate_auc_plot_image(results_df, compound_name, z, threshold, output_dir="output"):
    y_true = results_df["True Label"]
    y_scores = results_df["Intensity\n(cps)"]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    min_thresh = np.min(thresholds)
    max_thresh = np.max(thresholds)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {compound_name} (z={z})')
    plt.legend(loc='lower right')
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{compound_name}_roc_curve.png")
    plt.savefig(image_path)
    plt.close()

    return image_path, roc_auc, min_thresh, max_thresh

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    z = 1.1  # fixed for test

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_file = os.path.splitext(filename)[0] + "_results.xlsx"
            output_path = os.path.join(OUTPUT_FOLDER, output_file)

            print(f"🔄 Processing {filename}...")

            try:
                Compound_G, Compound_F = load_and_clean_data(input_path)
                cv_values = extract_cv(input_path)

                results_g, threshold_g = classify_samples_whole(Compound_G, cv_values["Compound_G"]["Cv90"], z)
                results_f, threshold_f = classify_samples_whole(Compound_F, cv_values["Compound_F"]["Cv90"], z)

                image_g, auc_g, min_thresh_g, max_thresh_g = generate_auc_plot_image(
                    results_g, "Compound_G", z, threshold_g, OUTPUT_FOLDER
                )
                image_f, auc_f, min_thresh_f, max_thresh_f = generate_auc_plot_image(
                    results_f, "Compound_F", z, threshold_f, OUTPUT_FOLDER
                )

                with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                    results_g.to_excel(writer, sheet_name="Compound G", index=False)
                    results_f.to_excel(writer, sheet_name="Compound F", index=False)

                    summary = pd.DataFrame({
                        "Compound": ["Compound G", "Compound F"],
                        "Z Value": [z, z],
                        "Threshold": [threshold_g, threshold_f],
                        "AUC": [auc_g, auc_f],
                        "Min ROC Threshold": [min_thresh_g, min_thresh_f],
                        "Max ROC Threshold": [max_thresh_g, max_thresh_f]
                    })
                    summary.to_excel(writer, sheet_name="Summary", index=False)

                print(f"✅ Saved to: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

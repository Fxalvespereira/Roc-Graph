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

def extract_cv_lod(filepath):
    sheet = pd.read_excel(filepath, sheet_name="Sheet1", header=None)
    return {
        "Compound_G": {"Cv90": sheet.iloc[1, 2]},
        "Compound_F": {"Cv90": sheet.iloc[1, 8]}
    }

def classify_samples_whole(df, cv90, z):
    threshold = cv90 * z
    results = df[["Sample ID", "Concentration\n(ng)", "Intensity\n(cps)"]].copy()
    results["Above Threshold"] = df["Intensity\n(cps)"] > threshold
    results["True Label"] = (df["Concentration\n(ng)"] != 0).astype(int)
    return results

def generate_auc_plot_image(results_df, compound_name, output_dir="output"):
    y_true = results_df["True Label"]
    y_scores = results_df["Intensity\n(cps)"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {compound_name}')
    plt.legend(loc='lower right')
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"{compound_name}_roc_curve.png")
    plt.savefig(image_path)
    plt.close()
    return image_path

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Uncomment below for real user input
    # try:
    #     z = float(input("Enter the z value to use for threshold calculation (e.g. 1.1): "))
    # except ValueError:
    #     print("❌ Invalid input for z. Please enter a numeric value.")
    #     return

    z = 1.1  # fallback for environments without input()

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            input_path = os.path.join(INPUT_FOLDER, filename)

            print(f"🔄 Processing {filename}...")

            try:
                Compound_G, Compound_F = load_and_clean_data(input_path)
                results = extract_cv_lod(input_path)

                results_g = classify_samples_whole(Compound_G, results["Compound_G"]["Cv90"], z)
                results_f = classify_samples_whole(Compound_F, results["Compound_F"]["Cv90"], z)

                generate_auc_plot_image(results_g, "Compound_G", OUTPUT_FOLDER)
                generate_auc_plot_image(results_f, "Compound_F", OUTPUT_FOLDER)

                print(f"✅ Saved plots for: {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

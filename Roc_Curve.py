import os
import pandas as pd
from sklearn.metrics import roc_curve
import xlsxwriter

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def num2col(n):
    s = ''
    while True:
        n, r = divmod(n, 26)
        s = chr(65 + r) + s
        if n == 0:
            break
        n -= 1
    return s

def autodetect_channels(sheetpath):
    tmp = pd.read_excel(sheetpath, header=None, nrows=6)
    header_row = tmp.iloc[5]
    n_cols = header_row.last_valid_index() + 1

    channels = []
    col_start, channel_idx = 0, 1

    while col_start < n_cols:
        end_col = min(col_start + 6, n_cols - 1)
        if (end_col - col_start) < 6:
            break  # Ignore incomplete channels at the end
        cols = f"{num2col(col_start)}:{num2col(end_col)}"
        skiprows = [0, 4] if channel_idx == 1 else [0, 1]
        channels.append((f"Channel {channel_idx}", skiprows, cols, col_start + 2))
        col_start += 7
        channel_idx += 1

    return channels

def load_and_clean_data(filepath, channels):
    dfs = {}
    for name, skip, cols, _ in channels:
        df = pd.read_excel(filepath, skiprows=skip, usecols=cols, names=[
            "Sample ID", "Concentration (ng)", "Intensity (cps)", "Mass (m/z)",
            "Mass 15 Intensity (cps)", "Mass 18 Intensity (cps)", "Mass 18 Ratio"
        ])
        df["Concentration (ng)"] = pd.to_numeric(df["Concentration (ng)"], errors="coerce")
        df["Intensity (cps)"] = pd.to_numeric(df["Intensity (cps)"], errors="coerce")
        dfs[name] = df.dropna(subset=["Sample ID", "Concentration (ng)", "Intensity (cps)"]).iloc[1:].reset_index(drop=True)
    return dfs

def extract_cv(filepath, channels):
    s = pd.read_excel(filepath, sheet_name=0, header=None)
    return {name: s.iloc[1, cv_col] for name, _, _, cv_col in channels}

def classify(df, cv, z):
    threshold = cv * z
    def label(row):
        if row["Concentration (ng)"] != 0:
            return "True Positive" if row["Intensity (cps)"] > threshold else "False Negative"
        else:
            return "False Positive" if row["Intensity (cps)"] > threshold else "True Negative"
    df = df.copy()
    df["Classification"] = df.apply(label, axis=1)
    df["True Label"] = df["Classification"].isin(["True Positive", "False Negative"]).astype(int)
    return df

def count(df, prefix):
    vals = df['Classification'].value_counts()
    return {f"{prefix} {k}": vals.get(k, 0) for k in ["True Positive", "False Positive", "False Negative", "True Negative"]}

def add_roc(workbook, worksheet, rocs, counts, cvs, z, channel_names):
    roc_sheet = workbook.add_worksheet("ROC Data")
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']

    for idx, name in enumerate(channel_names):
        roc_sheet.write(0, idx*2, f"FPR ({name})")
        roc_sheet.write(0, idx*2+1, f"TPR ({name})")
        fpr, tpr = rocs[name]
        for i, (f, t) in enumerate(zip(fpr, tpr)):
            roc_sheet.write(i+1, idx*2, f)
            roc_sheet.write(i+1, idx*2+1, t)

    chart = workbook.add_chart({'type':'scatter', 'subtype':'smooth'})
    for i, name in enumerate(channel_names):
        fpr, tpr = rocs[name]
        chart.add_series({
            'name': name,
            'categories': ['ROC Data', 1, i*2, len(fpr), i*2],
            'values': ['ROC Data', 1, i*2+1, len(tpr), i*2+1],
            'line': {'color': colors[i % len(colors)]}
        })
    chart.set_x_axis({'name': 'False Positive Rate'})
    chart.set_y_axis({'name': 'True Positive Rate'})
    worksheet.insert_chart('B10', chart)

def process_file(filepath, z):
    channels = autodetect_channels(filepath)
    dfs = load_and_clean_data(filepath, channels)
    cvs = extract_cv(filepath, channels)

    classified = {name: classify(dfs[name], cvs[name], z) for name, *_ in channels}
    counts = {name: count(classified[name], name) for name in classified}
    rocs = {name: roc_curve(classified[name]["True Label"], classified[name]["Intensity (cps)"])[:2] for name in classified}

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(filepath).replace('.xlsx', '_results.xlsx'))

    with pd.ExcelWriter(output_path, engine='xlsxwriter', engine_kwargs={'options':{'nan_inf_to_errors': True}}) as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Combined")
        writer.sheets["Combined"] = worksheet

        offset = 0
        for name in classified:
            worksheet.write(0, offset, name)
            for col_num, col_name in enumerate(classified[name].columns):
                worksheet.write(1, offset + col_num, col_name)
            for r, row in classified[name].iterrows():
                for c, v in enumerate(row):
                    if pd.isna(v) or v in [float('inf'), float('-inf')]:
                        worksheet.write(r+2, offset+c, "")
                    else:
                        worksheet.write(r+2, offset+c, v)
            offset += len(classified[name].columns) + 2

        add_roc(workbook, worksheet, rocs, counts, cvs, z, list(classified.keys()))

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    z = 1.1
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".xlsx") and not file.startswith("~$"):
            print(f"Processing {file}...")
            try:
                process_file(os.path.join(INPUT_FOLDER, file), z=z)
                print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()

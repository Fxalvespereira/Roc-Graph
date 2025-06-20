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
    n_cols = len(header_row)
    channels = []
    channel_idx = 1
    for col_start in range(n_cols):
        cell = header_row.iloc[col_start]
        if pd.isna(cell): continue
        if str(cell).strip().lower().startswith("sample id"):
            col_end = col_start + 6
            if col_end >= n_cols: col_end = n_cols-1
            cols = f"{num2col(col_start)}:{num2col(col_end)}"
            skiprows = [0, 4] if len(channels) == 0 else [0, 1]
            cv_col = col_start + 2
            channels.append((f"Channel {channel_idx}", skiprows, cols, cv_col))
            channel_idx += 1
    return channels

def load_and_clean_data(filepath, channels):
    dfs = {}
    for name, skip, cols, _ in channels:
        try:
            df = pd.read_excel(filepath, skiprows=skip, usecols=cols, names=[
                "Sample ID", "Concentration (ng)", "Intensity (cps)", "Mass (m/z)",
                "Mass 15 Intensity (cps)", "Mass 18 Intensity (cps)", "Mass 18 Ratio"
            ])
            df["Concentration (ng)"] = pd.to_numeric(df["Concentration (ng)"], errors="coerce")
            df["Intensity (cps)"] = pd.to_numeric(df["Intensity (cps)"], errors="coerce")
            # Remove completely empty rows, then skip the top row which is a repeat of headers
            data = df.dropna(subset=["Sample ID", "Concentration (ng)", "Intensity (cps)"])
            if len(data) > 1:
                dfs[name] = data.iloc[1:].reset_index(drop=True)
        except Exception as e:
            print(f"Error loading channel {name}: {e}")
    return dfs

def extract_cv(filepath, channels):
    s = pd.read_excel(filepath, sheet_name=0, header=None)
    cvs = {}
    for name, _, _, cv_col in channels:
        try:
            cv = s.iloc[1, cv_col]
            if pd.isna(cv) or cv in [float("nan"), float("inf"), float("-inf")]:
                cv = 0
        except Exception:
            cv = 0
        cvs[name] = cv
    return cvs

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
    any_series = False

    for idx, name in enumerate(channel_names):
        fpr, tpr = rocs[name]
        if len(fpr) > 1 and len(tpr) > 1:
            any_series = True
            roc_sheet.write(0, idx*2, f"FPR ({name})")
            roc_sheet.write(0, idx*2+1, f"TPR ({name})")
            for i, (f, t) in enumerate(zip(fpr, tpr)):
                roc_sheet.write(i+1, idx*2, f)
                roc_sheet.write(i+1, idx*2+1, t)
        else:
            roc_sheet.write(0, idx*2, f"FPR ({name})")
            roc_sheet.write(0, idx*2+1, f"TPR ({name})")
    # ROC chart
    chart = workbook.add_chart({'type':'scatter', 'subtype':'smooth'})
    for i, name in enumerate(channel_names):
        fpr, tpr = rocs[name]
        if len(fpr) > 1 and len(tpr) > 1:
            chart.add_series({
                'name': name,
                'categories': ['ROC Data', 1, i*2, len(fpr), i*2],
                'values': ['ROC Data', 1, i*2+1, len(tpr), i*2+1],
                'line': {'color': colors[i % len(colors)]}
            })
    if any_series:
        chart.set_x_axis({'name': 'False Positive Rate'})
        chart.set_y_axis({'name': 'True Positive Rate'})
        roc_sheet.insert_chart('J2', chart)

def process_file(filepath, z):
    channels = autodetect_channels(filepath)
    if not channels:
        print(f"Could not detect any channels in {filepath}")
        return
    dfs = load_and_clean_data(filepath, channels)
    cvs = extract_cv(filepath, channels)
    if not dfs:
        print(f"No data to process in {filepath}")
        return

    classified = {name: classify(dfs[name], cvs[name], z) for name in dfs}
    counts = {name: count(classified[name], name) for name in classified}
    rocs = {name: roc_curve(classified[name]["True Label"], classified[name]["Intensity (cps)"])[:2]
            for name in classified}

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(filepath).replace('.xlsx', '_results.xlsx'))

    with pd.ExcelWriter(output_path, engine='xlsxwriter', engine_kwargs={'options':{'nan_inf_to_errors': True}}) as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Combined")
        writer.sheets["Combined"] = worksheet

        # Color palette for up to 8 channels
        colors = ['#D9E1F2', '#FCE4D6', '#E2EFDA', '#FFF2CC', '#F8CBAD', '#DEEBF7', '#F4CCCC', '#D0E0E3']
        bold_formats = [workbook.add_format({'bold': True, 'bg_color': c, 'font_size': 13, 'align':'center'}) for c in colors]
        normal_formats = [workbook.add_format({'bg_color': c}) for c in colors]

        row = 0
        for idx, name in enumerate(classified):
            color_bold = bold_formats[idx % len(bold_formats)]
            color_normal = normal_formats[idx % len(normal_formats)]

            worksheet.merge_range(row, 0, row, len(classified[name].columns)-1, name, color_bold)
            row += 1

            for col_num, col_name in enumerate(classified[name].columns):
                worksheet.write(row, col_num, col_name, color_bold)
            row += 1

            for r, record in classified[name].iterrows():
                for c, v in enumerate(record):
                    if pd.isna(v) or v in [float('inf'), float('-inf')]:
                        worksheet.write(row, c, "", color_normal)
                    else:
                        worksheet.write(row, c, v, color_normal)
                row += 1

            row += 1

            worksheet.write(row, 0, f"{name} Confusion Matrix", color_bold)
            worksheet.write_row(row + 1, 0, ["True Positive", "False Positive", "False Negative", "True Negative"], color_bold)
            worksheet.write_row(row + 2, 0, [
                counts[name].get(f"{name} True Positive", 0),
                counts[name].get(f"{name} False Positive", 0),
                counts[name].get(f"{name} False Negative", 0),
                counts[name].get(f"{name} True Negative", 0)
            ], color_normal)
            row += 4

            worksheet.write(row, 0, "Critical Value", color_bold)
            worksheet.write(row, 1, cvs[name])
            worksheet.write(row + 1, 0, "Z Value", color_bold)
            worksheet.write(row + 1, 1, z)
            row += 3

            row += 1

        for col in range(0, 30):
            worksheet.set_column(col, col, 22)

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

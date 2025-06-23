import os
import pandas as pd
from sklearn.metrics import roc_curve
import xlsxwriter

# Set folder names for input and output
INPUT_FOLDER, OUTPUT_FOLDER = "input", "output"

def load_data(path):
    """
    Loads Compound G and Compound F data from an Excel file.
    - Reads columns A:G for Compound G, skipping specific header rows.
    - Reads columns A, H:M for Compound F, skipping other header rows.
    - Drops NA, resets index, ensures correct types for numeric columns.
    Returns: {"G": dataframe, "F": dataframe}
    """
    # Column headers for both compounds
    names = ["Sample ID","Concentration (ng)","Intensity (cps)","Mass (m/z)",
             "Mass 15 Intensity (cps)","Mass 18 Intensity (cps)","Mass 18 Ratio"]
    # Load Compound G data (columns A:G, skip first and row 4)
    dfg = pd.read_excel(path, skiprows=[0,4], nrows=52, usecols="A:G", names=names)
    # Load Compound F data (columns A and H:M, skip first and row 1)
    dff = pd.read_excel(path, skiprows=[0,1], nrows=52, usecols="A,H:M", names=names)
    # Convert necessary columns to numeric for calculation
    for df in [dfg, dff]:
        df["Concentration (ng)"] = pd.to_numeric(df["Concentration (ng)"], errors="coerce")
        df["Intensity (cps)"] = pd.to_numeric(df["Intensity (cps)"], errors="coerce")
    # Drop NAs, skip first row (typically header), and return both
    return {
        "G": dfg.dropna().iloc[1:].reset_index(drop=True),
        "F": dff.dropna().iloc[1:].reset_index(drop=True)
    }

def extract_cv(path):
    """
    Extracts Cv90 (critical value) for each channel from the input Excel header rows.
    - Cv90 for G is cell (1,2) and for F is (1,8).
    Returns: {"G": value, "F": value}
    """
    s = pd.read_excel(path, sheet_name="Sheet1", header=None)
    return {"G": s.iloc[1,2], "F": s.iloc[1,8]}

def classify(df, cv, z):
    """
    Classifies each row as TP, FP, FN, or TN based on:
    - Whether it's a blank (Concentration == 0) or sample (Concentration != 0)
    - Whether Intensity > threshold (cv * z) or not
    Adds columns for 'Classification', 'True Label' (for ROC), and 'Above Threshold'.
    Returns the dataframe with new columns.
    """
    t = cv * z  # Threshold for classification
    df = df.copy()
    # Classification logic
    def label(row):
        if row["Concentration (ng)"] != 0:
            return "True Positive" if row["Intensity (cps)"] > t else "False Negative"
        return "False Positive" if row["Intensity (cps)"] > t else "True Negative"
    df["Classification"] = df.apply(label, axis=1)
    # True Label (for ROC): 1 for any sample (TP or FN), 0 for blank (FP or TN)
    df["True Label"] = df["Classification"].isin(["True Positive","False Negative"]).astype(int)
    # Above Threshold (for ROC): 1 if above, 0 if not
    df["Above Threshold"] = df["Classification"].isin(["True Positive","False Positive"]).astype(int)
    return df

def confusion_counts(df, ch):
    """
    Counts confusion matrix components (TP, FP, FN, TN) for a channel.
    Prefixes with channel name ('G' or 'F').
    Returns a dict with counts.
    """
    v = df['Classification'].value_counts()
    return {f"{ch} {x}": v.get(x,0) for x in ["True Positive","False Positive","False Negative","True Negative"]}

def add_roc(wb, ws, fprg, tprg, fprf, tprf, cg, cf, cvg, cvf, z, fpr_comb, tpr_comb):
    """
    Adds ROC data and summary/confusion matrix tables to Excel workbook.
    Also adds editable ROC charts for each channel and for the combined ROC.
    """
    sheet = wb.add_worksheet("ROC Data")
    # Formats for channel highlighting
    fmt_g, fmt_f, fmt_c = wb.add_format({'bold':1,'bg_color':'#D9E1F2'}), wb.add_format({'bold':1,'bg_color':'#FCE4D6'}), wb.add_format({'bold':1,'bg_color':'#FFD1DC'})
    # Write individual ROC data
    sheet.write_row(0,0,["FPR (G)","TPR (G)","","FPR (F)","TPR (F)"],fmt_g)
    [sheet.write(i+1,0,fg) or sheet.write(i+1,1,tg) for i,(fg,tg) in enumerate(zip(fprg,tprg))]
    [sheet.write(i+1,3,ff) or sheet.write(i+1,4,tf) for i,(ff,tf) in enumerate(zip(fprf,tprf))]
    # Write combined ROC data
    sheet.write(0, 7, "FPR (Combined)", fmt_c)
    sheet.write(0, 8, "TPR (Combined)", fmt_c)
    for i,(f,t) in enumerate(zip(fpr_comb, tpr_comb)): 
        sheet.write(i+1,7,f); sheet.write(i+1,8,t)
    # Write confusion matrix for G and F
    sheet.write_row(1,6,["Channel","TP","FP","FN","TN"],wb.add_format({'bold':1}))
    sheet.write_row(2,6,["Channel G",cg.get("G True Positive",0),cg.get("G False Positive",0),cg.get("G False Negative",0),cg.get("G True Negative",0)],fmt_g)
    sheet.write_row(3,6,["Channel F",cf.get("F True Positive",0),cf.get("F False Positive",0),cf.get("F False Negative",0),cf.get("F True Negative",0)],fmt_f)
    # Z value and Cv90s
    sheet.write(5,6,"Z Value",wb.add_format({'bold':1})); sheet.write(5,7,z)
    sheet.write(6,6,"Critical Value (G)",fmt_g); sheet.write(6,7,cvg)
    sheet.write(7,6,"Critical Value (F)",fmt_f); sheet.write(7,7,cvf)
    sheet.set_column(0,10,25)
    # Editable ROC chart for each channel
    chart = wb.add_chart({'type':'scatter','subtype':'smooth_with_markers'})
    chart.add_series({'name':'Channel G','categories':['ROC Data',1,0,len(fprg),0],'values':['ROC Data',1,1,len(tprg),1],'line':{'color':'blue'}})
    chart.add_series({'name':'Channel F','categories':['ROC Data',1,3,len(fprf),3],'values':['ROC Data',1,4,len(tprf),4],'line':{'color':'red'}})
    chart.set_title({'name':'ROC Curve'})
    chart.set_x_axis({'name':'False Positive Rate'})
    chart.set_y_axis({'name':'True Positive Rate'})
    ws.insert_chart('G2',chart)
    # Editable combined ROC chart
    chart2 = wb.add_chart({'type':'scatter','subtype':'smooth_with_markers'})
    chart2.add_series({'name':'Combined Channels','categories':['ROC Data',1,7,len(fpr_comb),7],'values':['ROC Data',1,8,len(tpr_comb),8],'line':{'color':'purple'}})
    chart2.set_title({'name':'Combined ROC Curve (G+F)'})
    chart2.set_x_axis({'name':'False Positive Rate'})
    chart2.set_y_axis({'name':'True Positive Rate'})
    ws.insert_chart('N2',chart2)

def process_file(path, z):
    """
    Orchestrates loading, classifying, counting, ROC calculation, and Excel output
    for a single Excel data file.
    """
    d = load_data(path)      # Load and clean both compounds
    cv = extract_cv(path)    # Extract Cv90s from header
    # Classify G and F compounds using the user-provided Z value
    ch = {k: classify(d[k], cv[k], z) for k in ["G","F"]}
    cg, cf = confusion_counts(ch["G"],"G"), confusion_counts(ch["F"],"F")
    # ROC curve for each channel
    fprg, tprg, _ = roc_curve(ch["G"]["True Label"], ch["G"]["Intensity (cps)"])
    fprf, tprf, _ = roc_curve(ch["F"]["True Label"], ch["F"]["Intensity (cps)"])
    # Combined ROC curve (merge all true labels and scores)
    all_true = pd.concat([ch["G"]["True Label"], ch["F"]["True Label"]], ignore_index=True)
    all_scores = pd.concat([ch["G"]["Intensity (cps)"], ch["F"]["Intensity (cps)"]], ignore_index=True)
    mask = (~all_true.isna()) & (~all_scores.isna())
    fpr_comb, tpr_comb, _ = roc_curve(all_true[mask], all_scores[mask])
    # Output file name includes Z value
    fn = os.path.splitext(os.path.basename(path))[0]+f"_z{z}_results.xlsx"
    out = os.path.join(OUTPUT_FOLDER, fn)
    # Write combined Excel file
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        wb = writer.book; ws = wb.add_worksheet("Combined"); writer.sheets['Combined'] = ws
        fmt_g, fmt_f = wb.add_format({'bold':1,'bg_color':'#D9E1F2'}), wb.add_format({'bold':1,'bg_color':'#FCE4D6'})
        # Write G and F channel tables, side by side, with color formatting
        ws.write(0,0,"Channel G",fmt_g); ws.write(0,len(ch["G"].columns)+2,"Channel F",fmt_f)
        for k,df,fmt,ofs in zip(["G","F"],[ch["G"],ch["F"]],[fmt_g,fmt_f],[0,len(ch["G"].columns)+2]):
            ws.write_row(1,ofs,list(df.columns),fmt)
            for r,row in df.iterrows():
                ws.write_row(r+2,ofs,list(row))
        ws.set_column(0, len(ch["G"].columns)+len(ch["F"].columns)+2, 22)
        # Write summary confusion matrices for each channel
        base = max(len(ch["G"]),len(ch["F"]))+4
        for i,(label,val) in enumerate(cg.items()):
            ws.write(base+i,0,label,fmt_g); ws.write(base+i,1,val)
        for i,(label,val) in enumerate(cf.items()):
            ws.write(base+i,len(ch["G"].columns)+2,label,fmt_f); ws.write(base+i,len(ch["G"].columns)+3,val)
        # Add ROC data and charts
        add_roc(wb, ws, fprg, tprg, fprf, tprf, cg, cf, cv["G"], cv["F"], z, fpr_comb, tpr_comb)

def main():
    """
    Batch processes all .xlsx files in the input folder.
    Prompts user for Z value, creates output folder, and writes progress.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    try:
        z = float(input("Enter the Z value (e.g. 1.1): "))
    except Exception:
        print("Invalid value for Z. Using 1.1.")
        z = 1.1
    for f in os.listdir(INPUT_FOLDER):
        if f.endswith(".xlsx") and not f.startswith("~$"):
            print(f"Processing {f} with Z={z}...")
            process_file(os.path.join(INPUT_FOLDER, f), z=z)

if __name__ == "__main__":
    main()

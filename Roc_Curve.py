import os
import pandas as pd
from sklearn.metrics import roc_curve
import xlsxwriter

INPUT_FOLDER, OUTPUT_FOLDER = "input", "output"

def load_and_clean_data(filepath):
    # Returns a dict: {"G": df, "F": df}
    def read(cols, skips): 
        return pd.read_excel(filepath, skiprows=skips, nrows=52, usecols=cols, 
            names=["Sample ID", "Concentration (ng)", "Intensity (cps)", "Mass (m/z)", 
                   "Mass 15 Intensity (cps)", "Mass 18 Intensity (cps)", "Mass 18 Ratio"])
    dfg = read("A:G", [0,4])
    dff = read("A,H:M", [0,1])
    for df in [dfg, dff]:
        df["Concentration (ng)"] = pd.to_numeric(df["Concentration (ng)"], errors="coerce")
        df["Intensity (cps)"] = pd.to_numeric(df["Intensity (cps)"], errors="coerce")
    return { 
        "G": dfg.dropna().iloc[1:].reset_index(drop=True), 
        "F": dff.dropna().iloc[1:].reset_index(drop=True) 
    }

def extract_cv(filepath): 
    s = pd.read_excel(filepath, sheet_name="Sheet1", header=None)
    return { "G": s.iloc[1,2], "F": s.iloc[1,8] }

def classify(df, cv, z):
    t = cv*z
    def lab(row):
        if row["Concentration (ng)"] != 0:
            return "True Positive" if row["Intensity (cps)"] < t else "False Negative"
        return "False Positive" if row["Intensity (cps)"] < t else "True Negative"
    df = df.copy()
    df["Classification"] = df.apply(lab, axis=1)
    df["True Label"] = df["Classification"].isin(["True Positive","False Negative"]).astype(int)
    df["Above Threshold"] = df["Classification"].isin(["True Positive","False Positive"]).astype(int)
    return df

def count(df, p): 
    v = df['Classification'].value_counts()
    return {f"{p} {k}":v.get(k,0) for k in ["True Positive","False Positive","False Negative","True Negative"]}

def add_roc(workbook, worksheet, fprg, tprg, fprf, tprf, cg, cf, cvg, cvf, z):
    sheet = workbook.add_worksheet("ROC Data")
    gf, ff = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}), workbook.add_format({'bold': True, 'bg_color': '#FCE4D6'})
    sheet.write_row(0,0,["False Positive Rate (Channel G)","True Positive Rate (Channel G)","","False Positive Rate (Channel F)","True Positive Rate (Channel F)"],gf)
    [sheet.write(i+1,0,fg) or sheet.write(i+1,1,tg) for i,(fg,tg) in enumerate(zip(fprg,tprg))]
    [sheet.write(i+1,3,ffv) or sheet.write(i+1,4,tfv) for i,(ffv,tfv) in enumerate(zip(fprf,tprf))]
    sheet.write(0,6,"Confusion Matrix",workbook.add_format({'bold':True}))
    sheet.write_row(1,6,["Channel","True Positive","False Positive","False Negative","True Negative"],workbook.add_format({'bold':True}))
    sheet.write_row(2,6,["Channel G",cg.get("G True Positive",0),cg.get("G False Positive",0),cg.get("G False Negative",0),cg.get("G True Negative",0)],gf)
    sheet.write_row(3,6,["Channel F",cf.get("F True Positive",0),cf.get("F False Positive",0),cf.get("F False Negative",0),cf.get("F True Negative",0)],ff)
    sheet.write(5,6,"Z Value",workbook.add_format({'bold':True})); sheet.write(5,7,z)
    sheet.write(6,6,"Critical Value (Channel G)",gf); sheet.write(6,7,cvg)
    sheet.write(7,6,"Critical Value (Channel F)",ff); sheet.write(7,7,cvf)
    sheet.set_column(0,10,25)
    chart = workbook.add_chart({'type':'scatter','subtype':'smooth_with_markers'})
    chart.add_series({'name':'Channel G','categories':['ROC Data',1,0,len(fprg),0],'values':['ROC Data',1,1,len(tprg),1],'marker':{'type':'circle'},'line':{'color':'blue'}})
    chart.add_series({'name':'Channel F','categories':['ROC Data',1,3,len(fprf),3],'values':['ROC Data',1,4,len(tprf),4],'marker':{'type':'square'},'line':{'color':'red'}})
    chart.set_title({'name':'ROC Curve'}); chart.set_x_axis({'name':'False Positive Rate'}); chart.set_y_axis({'name':'True Positive Rate'}); chart.set_legend({'position':'bottom'})
    worksheet.insert_chart('G2',chart)

def process_file(filepath, z):
    d = load_and_clean_data(filepath); cv = extract_cv(filepath)
    ch = {k: classify(d[k], cv[k], z) for k in ["G","F"]}
    cg, cf = count(ch["G"],"G"), count(ch["F"],"F")
    fprg, tprg, _ = roc_curve(ch["G"]["True Label"], ch["G"]["Intensity (cps)"])
    fprf, tprf, _ = roc_curve(ch["F"]["True Label"], ch["F"]["Intensity (cps)"])
    # Output filename includes the Z value!
    fn = os.path.splitext(os.path.basename(filepath))[0]+f"_z{z}_results.xlsx"
    out = os.path.join(OUTPUT_FOLDER, fn)
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        wb = writer.book; ws = wb.add_worksheet("Combined"); writer.sheets['Combined'] = ws
        gf, ff = wb.add_format({'bold':True,'bg_color':'#D9E1F2'}), wb.add_format({'bold':True,'bg_color':'#FCE4D6'})
        ws.write(0,0,"Channel G",gf); ws.write(0,len(ch["G"].columns)+2,"Channel F",ff)
        for k,df,fmt,ofs in zip(["G","F"],[ch["G"],ch["F"]],[gf,ff],[0,len(ch["G"].columns)+2]):
            [ws.write(1,ofs+c,col,fmt) for c,col in enumerate(df.columns)]
            for r,row in df.iterrows():
                [ws.write(r+2,ofs+c,v) for c,v in enumerate(row)]
        ws.set_column(0, len(ch["G"].columns)+len(ch["F"].columns)+2, 22)
        base = max(len(ch["G"]),len(ch["F"]))+4
        for i,(label,val) in enumerate(cg.items()):
            ws.write(base+i,0,label,gf); ws.write(base+i,1,val)
        for i,(label,val) in enumerate(cf.items()):
            ws.write(base+i,len(ch["G"].columns)+2,label,ff); ws.write(base+i,len(ch["G"].columns)+3,val)
        add_roc(wb, ws, fprg, tprg, fprf, tprf, cg, cf, cv["G"], cv["F"], z)

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    try:
        z = float(input("Enter the Z value (for example, 1.1): "))
    except Exception:
        print("Invalid value for Z. Using default Z=1.1")
        z = 1.1
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".xlsx") and not file.startswith("~$"):
            print(f"Processing {file} with Z={z}...")
            process_file(os.path.join(INPUT_FOLDER, file), z=z)

if __name__ == "__main__":
    main()

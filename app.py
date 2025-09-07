import gradio as gr, pandas as pd, numpy as np, joblib, os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

bundle = joblib.load("model_TCO_full.pkl")
model, scaler, le = bundle["model"], bundle["scaler"], bundle["label_encoder"]

def _norm(s):  # normalise header names
    return (str(s).lower().replace("θ","theta").replace("°","")
            .replace("(", "").replace(")", "").replace(".", "")
            .replace(" ", "").replace("-", ""))

def _pick_cols(df):
    theta_col=inten_col=None
    for c in df.columns:
        n=_norm(c)
        if n in {"2theta","2thetadegrees","twotheta","2thetadeg"}: theta_col=c
        if n in {"intensity","intensityau","counts","int"}: inten_col=c
    if theta_col is None and "2Theta (degrees)" in df.columns: theta_col="2Theta (degrees)"
    if inten_col is None and "Intensity" in df.columns: inten_col="Intensity"
    return theta_col, inten_col

def _read_any(file):
    path=getattr(file,"name",file)
    ext=os.path.splitext(path)[1].lower()
    sheets={}
    if ext in [".xlsx",".xls"]:
        for name,df in pd.read_excel(path, sheet_name=None).items():
            th,it=_pick_cols(df)
            if th and it:
                sheets[name]=df[[th,it]].rename(columns={th:"2theta", it:"intensity"})
    elif ext in [".csv",".txt"]:
        df=pd.read_csv(path)
        th,it=_pick_cols(df)
        if th and it: sheets["CSV"]=df[[th,it]].rename(columns={th:"2theta", it:"intensity"})
    if not sheets:
        raise ValueError("Fail mesti ada kolum 2theta & intensity (apa-apa sheet).")
    return sheets

def _features(theta,intensity,n=1000):
    idx=np.argsort(theta); theta=np.asarray(theta)[idx]; intensity=np.asarray(intensity)[idx]
    mask=np.isfinite(theta)&np.isfinite(intensity); theta, intensity = theta[mask], intensity[mask]
    grid=np.linspace(10,80,n)
    y=np.interp(grid, theta, intensity)
    max_i=float(np.max(y)); mean_i=float(np.mean(y)); std_i=float(np.std(y)); area=float(trapezoid(y,grid))
    peaks,_=find_peaks(y, height=max_i*0.8); peak_cnt=int(len(peaks))
    X=np.concatenate([y,[peak_cnt,max_i,mean_i,std_i,area]], dtype=float).reshape(1,-1)  # 505 dims
    return X, grid, y

def predict_file(file):
    try:
        sheets=_read_any(file)
        rows=[]; figs={}
        X_list=[]
        for name,df in sheets.items():
            X,_grid,_y=_features(df["2theta"].astype(float).values, df["intensity"].astype(float).values)
            X_list.append((name,X,df))
        # batch scale & predict
        X_stack=np.vstack([x for _,x,_ in X_list])
        X_scaled=scaler.transform(X_stack)
        y_idx=model.predict(X_scaled)
        labels=le.inverse_transform(y_idx)

        # optional probabilities if available
        prob=None
        if hasattr(model,"predict_proba"):
            prob = model.predict_proba(X_scaled)

        for i,(name,_,df) in enumerate(X_list):
            lbl=labels[i]
            rows.append({"sheet":name, "predicted_TCO":lbl, **({} if prob is None else {"Accuracy":float(np.max(prob[i]))})})
            # plot original curve
            fig,ax=plt.subplots()
            ax.plot(df["2theta"], df["intensity"], marker='o'); ax.grid(True)
            ax.set_title(f"{name} (Pred: {lbl})"); ax.set_xlabel("2θ (deg)"); ax.set_ylabel("Intensity (a.u.)")
            figs[name]=fig

        summary=pd.DataFrame(rows)
        out_path="predicted_output.csv"; summary.to_csv(out_path, index=False)

        # first sheet plot & dropdown
        first=list(sheets.keys())[0]
        return summary, out_path, gr.update(choices=list(sheets.keys()), value=first), figs[first]
    except Exception as e:
        return pd.DataFrame({"error":[str(e)]}), None, gr.update(choices=[], value=None), plt.figure()

def plot_sheet(file, sheet_name):
    if sheet_name is None: return plt.figure()
    sheets=_read_any(file)
    df=sheets[sheet_name]
    fig,ax=plt.subplots()
    ax.plot(df["2theta"], df["intensity"], marker='o'); ax.grid(True)
    ax.set_title(sheet_name); ax.set_xlabel("2θ (deg)"); ax.set_ylabel("Intensity (a.u.)")
    return fig

with gr.Blocks() as demo:
    gr.Markdown("## 🔬 TCO Material Predictor + XRD Visualizer (Excel/CSV, multi-sheet)")
    with gr.Row():
        with gr.Column():
            f=gr.File(label="Upload .xlsx / .csv")
            b=gr.Button("Predict", variant="primary")
        with gr.Column():
            table=gr.Dataframe(label="Ringkasan Prediksi (per sheet)")
            dl=gr.File(label="Download ringkasan (CSV)")
            sheet=gr.Dropdown(label="Pilih sheet untuk plot")
            plot=gr.Plot(label="Graf XRD")

    b.click(predict_file, inputs=f, outputs=[table, dl, sheet, plot])
    sheet.change(plot_sheet, inputs=[f, sheet], outputs=plot)

demo.launch()

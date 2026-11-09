"""Gradio web demo: pick a sample ECG (or upload your own) and see the model's predictions.

Run locally:   python app.py   ->   http://127.0.0.1:7860
Also works as a Hugging Face Space (see README, "Try the model").
"""

import json
import os
import shutil
import tempfile

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ptbxl.inference import ECGPredictor  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, "samples")
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXTRA_BEATS = ["none", "SVES", "VES", "SVES+VES"]

predictor = ECGPredictor(os.path.join(HERE, "model"))
with open(os.path.join(SAMPLES_DIR, "samples.json")) as f:
    SAMPLES = {s["name"]: s for s in json.load(f)}


# ---------------------------------------------------------------------------- #
def plot_ecg(signal):
    fig, axes = plt.subplots(6, 2, figsize=(12, 8), sharex=True)
    t = [i / 100 for i in range(signal.shape[0])]
    for k, ax in enumerate(axes.T.flatten()):
        ax.plot(t, signal[:, k], color="#c0392b", linewidth=0.8)
        ax.set_ylabel(LEADS[k], rotation=0, labelpad=15, fontsize=9)
        ax.grid(True, color="#f5b7b1", linewidth=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def load_upload(files):
    """Accept a WFDB pair (.hea + .dat) or a single .csv; returns a (1000, 12) signal."""
    paths = [f if isinstance(f, str) else f.name for f in files]
    csvs = [p for p in paths if p.lower().endswith(".csv")]
    if csvs:
        return predictor.read_csv(csvs[0])

    heas = [p for p in paths if p.lower().endswith(".hea")]
    if not heas:
        raise gr.Error("Upload a .hea + .dat pair (WFDB) or a .csv file.")
    # Gradio stores each upload in its own folder; WFDB needs .hea and .dat side by side.
    tmp = tempfile.mkdtemp()
    for p in paths:
        shutil.copy(p, tmp)
    record = os.path.splitext(os.path.basename(heas[0]))[0]
    return predictor.read_record(os.path.join(tmp, record))


def analyze(sample_name, files, age, sex, height, weight, pacemaker, extra_beats):
    truth = ""
    try:
        if files:
            signal = load_upload(files)
        elif sample_name:
            s = SAMPLES[sample_name]
            signal = predictor.read_record(os.path.join(SAMPLES_DIR, s["record"]))
            truth = f"**Cardiologist label for this sample:** {', '.join(s['true_labels'])}"
        else:
            raise gr.Error("Pick a sample or upload an ECG.")
    except ValueError as e:
        raise gr.Error(str(e))

    results = predictor.predict(
        signal,
        age=age, sex=None if sex == "unknown" else sex, height=height, weight=weight,
        pacemaker=pacemaker, extra_beats=None if extra_beats == "none" else extra_beats,
    )
    top = {f"{r['code']}: {r['description']}": r["probability"] for r in results[:8]}
    above = [r["code"] for r in results if r["probability"] >= 0.5]
    summary = f"**Predicted (p ≥ 0.5):** {', '.join(above) or 'none'}"
    if truth:
        summary += "  \n" + truth
    return plot_ecg(signal), top, summary


def fill_from_sample(sample_name):
    s = SAMPLES.get(sample_name)
    if not s:
        return [gr.update()] * 6
    return (
        s["age"], s["sex"], s["height"], s["weight"], s["pacemaker"],
        s["extra_beats"] if s["extra_beats"] in EXTRA_BEATS else "none",
    )


# ---------------------------------------------------------------------------- #
with gr.Blocks(title="PTB-XL ECG classifier") as demo:
    gr.Markdown(
        "# 12-lead ECG diagnosis demo\n"
        "A small 1D-CNN trained on [PTB-XL](https://physionet.org/content/ptb-xl/) that predicts "
        "44 diagnostic statements from a 10-second 12-lead ECG plus basic patient info "
        "(test macro AUROC 0.91).\n\n"
        "⚠️ **Portfolio project, not a medical device.** Do not use it for any clinical decision."
    )
    with gr.Row():
        with gr.Column(scale=1):
            sample = gr.Dropdown(list(SAMPLES), value=list(SAMPLES)[0], label="Try a sample ECG from the test set")
            files = gr.File(
                label="…or upload your own: WFDB .hea + .dat, or a CSV (12 columns, 100 Hz, ≥ 10 s)",
                file_count="multiple", file_types=[".hea", ".dat", ".csv"],
            )
            with gr.Accordion("Patient info (optional)", open=True):
                age = gr.Number(label="Age")
                sex = gr.Radio(["male", "female", "unknown"], value="unknown", label="Sex")
                with gr.Row():
                    height = gr.Number(label="Height (cm)")
                    weight = gr.Number(label="Weight (kg)")
                pacemaker = gr.Checkbox(label="Pacemaker")
                extra_beats = gr.Dropdown(EXTRA_BEATS, value="none", label="Extra beats")
            button = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="ECG")
            summary = gr.Markdown()
            labels = gr.Label(num_top_classes=8, label="Top predictions")

    meta_inputs = [age, sex, height, weight, pacemaker, extra_beats]
    sample.change(fill_from_sample, sample, meta_inputs)
    button.click(analyze, [sample, files, *meta_inputs], [plot, labels, summary])
    demo.load(fill_from_sample, sample, meta_inputs)


if __name__ == "__main__":
    demo.launch()

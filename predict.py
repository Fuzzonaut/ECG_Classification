"""Predict diagnoses for one ECG from the command line.

Examples:
    python predict.py samples/00009_lr --age 47 --sex female       # WFDB record (.hea + .dat)
    python predict.py my_ecg.csv --fs 500 --age 60 --sex male       # CSV: 12 columns, one row per sample
    python predict.py --list-samples                                # show the bundled demo ECGs

Not a medical device. For demonstration only.
"""

import argparse
import json
import os

from ptbxl.inference import ECGPredictor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", nargs="?", help="WFDB record path (without extension) or a .csv file")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--fs", type=float, default=100, help="sampling rate of a CSV input (Hz)")
    p.add_argument("--age", type=float)
    p.add_argument("--sex", choices=["male", "female"])
    p.add_argument("--height", type=float, help="cm")
    p.add_argument("--weight", type=float, help="kg")
    p.add_argument("--pacemaker", action="store_true")
    p.add_argument("--extra-beats", choices=["SVES", "VES", "SVES+VES"])
    p.add_argument("--top", type=int, default=5, help="number of labels to show")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--list-samples", action="store_true")
    args = p.parse_args()

    if args.list_samples:
        with open(os.path.join("samples", "samples.json")) as f:
            for s in json.load(f):
                print(f"samples/{s['record']:<10}  age {s['age']}, {s['sex']:<6}  true: {s['name']}")
        return
    if not args.record:
        p.error("give a record path, or use --list-samples")

    predictor = ECGPredictor(args.model_dir)
    if args.record.lower().endswith(".csv"):
        signal = predictor.read_csv(args.record, fs=args.fs)
    else:
        signal = predictor.read_record(os.path.splitext(args.record)[0])

    results = predictor.predict(
        signal, age=args.age, sex=args.sex, height=args.height, weight=args.weight,
        pacemaker=args.pacemaker, extra_beats=args.extra_beats,
    )

    positives = [r["code"] for r in results if r["probability"] >= args.threshold]
    print(f"Predicted (p >= {args.threshold}): {', '.join(positives) or 'none'}\n")
    print(f"Top {args.top}:")
    for r in results[: args.top]:
        bar = "#" * round(r["probability"] * 30)
        print(f"  {r['code']:<7} {r['probability']:6.1%}  {bar:<30}  {r['description']}")
    print("\nNot a medical device. For demonstration only.")


if __name__ == "__main__":
    main()

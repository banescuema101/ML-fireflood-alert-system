# lst/train_model_on_lst.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

FEATURES = ["LST_K","med_k","mad_k","mean_k","std_k","zmad_k"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="lst_fire_model.pkl")
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    X = df[FEATURES]; y = df["label"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, max_depth=18, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:,1]
    print("ROC AUC:", roc_auc_score(yte, prob))
    print(classification_report(yte, (prob>=0.5).astype(int), digits=3))

    joblib.dump({"model": clf, "features": FEATURES}, args.out)
    print(f">> Model salvat: {args.out}")

if __name__ == "__main__":
    main()

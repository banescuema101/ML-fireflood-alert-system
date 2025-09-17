# lst/train_model_on_lst.py
# Script for training an ML model (Random Forest) on LST pseudo-label data

import os, sys
# Add parent directory to sys.path to be able to import from utils if needed
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# List of features used for training the model
FEATURES = ["LST_K","med_k","mad_k","mean_k","std_k","zmad_k"]

def main():
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)               # input training file (parquet)
    ap.add_argument("--out", default="lst_fire_model.pkl")  # output file for saved model
    ap.add_argument("--test-size", type=float, default=0.25) # proportion of the test set
    args = ap.parse_args()

    # Read training data
    df = pd.read_parquet(args.train)  # load prepared dataset
    X = df[FEATURES]                  # feature matrix
    y = df["label"].astype(int)       # label vector (0/1)

    # Split train/test
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=y   # keep class distribution
    )

    # Define and train model - Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,      # number of trees
        max_depth=18,          # maximum depth of trees
        class_weight="balanced", # automatic balancing of classes (good for imbalanced datasets)
        n_jobs=-1,             # use all CPU cores
        random_state=42
    )
    clf.fit(Xtr, ytr)           # train model on training set

    # Evaluate on test set
    prob = clf.predict_proba(Xte)[:,1]  # probability for positive class
    print("ROC AUC:", roc_auc_score(yte, prob))  # ROC AUC score
    # Binary classification with threshold 0.3 and full report
    print(classification_report(yte, (prob>=0.3).astype(int), digits=3))

    # Save trained model
    joblib.dump({"model": clf, "features": FEATURES}, args.out)
    print(f">> Model saved: {args.out}")

if __name__ == "__main__":
    main()

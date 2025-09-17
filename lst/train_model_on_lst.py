# lst/train_model_on_lst.py
# Script pentru antrenarea unui model ML (Random Forest) pe date LST pseudo-label

import os, sys
# Adaugam directorul parinte in sys.path pentru a putea importa din utils daca e nevoie
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Lista feature-elor folosite pentru antrenarea modelului
FEATURES = ["LST_K","med_k","mad_k","mean_k","std_k","zmad_k"]

def main():
    # Parsare argumente
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)               # fisierul de input cu date de train (parquet)
    ap.add_argument("--out", default="lst_fire_model.pkl")  # fisierul de output pentru modelul salvat
    ap.add_argument("--test-size", type=float, default=0.25) # proportia setului de test
    args = ap.parse_args()

    # Citire date antrenament
    df = pd.read_parquet(args.train)  # incarcam dataset-ul pregatit
    X = df[FEATURES]                  # matricea de feature-uri
    y = df["label"].astype(int)       # vectorul de label-uri (0/1)

    # Split train/test
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=y   # pastram distributia claselor
    )

    # partea de definire si antrenare model - Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,      # nr. de arbori
        max_depth=18,          # adancimea maxima a arborilor
        class_weight="balanced", # balansare automata a claselor (bun pe dataseturi dezechilibrate)
        n_jobs=-1,             # foloseste toate core-urile CPU
        random_state=42
    )
    clf.fit(Xtr, ytr)           # antrenam modelul pe setul de train

    # evaluare pe setul de test
    prob = clf.predict_proba(Xte)[:,1]  # probabilitatea pentru clasa pozitiva
    print("ROC AUC:", roc_auc_score(yte, prob))  # scor AUC ROC
    # clasificare binara cu prag 0.5 si raport complet
    print(classification_report(yte, (prob>=0.3).astype(int), digits=3))

    # Salvare model antrenat
    joblib.dump({"model": clf, "features": FEATURES}, args.out)
    print(f">> Model salvat: {args.out}")

if __name__ == "__main__":
    main()

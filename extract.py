# extract.py.

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# 1) Încarci datele și alegi variabilele
ds = xr.open_dataset("/home/banescuema/ROSPIN/ML-fireflood-alert-system/Dataset/Sentinel-3-dataset-greece-august-2023/S3B_SL_2_FRP____20250811T090721_20250811T091221_20250811T111101_0299_109_378______MAR_O_NR_003.SEN3/FRP_MWIR1km_standard.nc")
vars_needed = ['MWIR_Fire_pixel_BT','S8_Fire_pixel_BT','FRP_MWIR','TCWV','solar_zenith','sat_zenith','confidence_MWIR']
df = ds[vars_needed].to_dataframe().reset_index()

# 2) Curățare rapidă
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=vars_needed)

# 3) Label (ajustează pragul dacă vrei mai puține FP / mai multe TP)
df['label'] = (df['confidence_MWIR'] > 80).astype(int)

features = ['MWIR_Fire_pixel_BT','S8_Fire_pixel_BT','FRP_MWIR','TCWV','solar_zenith','sat_zenith']
X = df[features]
y = df['label']

# 4) Train/test stratificat (dacă e dezechilibru, ajută mult)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 5) Model (DT simplu, ușor de explicat)
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 6) Evaluare
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

# 7) Salvezi modelul
joblib.dump(model, "fire_detection_tree.pkl")
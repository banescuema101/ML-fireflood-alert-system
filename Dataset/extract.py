import xarray as xr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_text
import joblib
import matplotlib.pyplot as plt

ds = xr.open_dataset("/home/banescuema/ROSPIN/ML-fireflood-alert-system/Dataset/Sentinel-3-dataset-greece-august-2023/S3B_SL_2_FRP____20250811T090721_20250811T091221_20250811T111101_0299_109_378______MAR_O_NR_003.SEN3/FRP_MWIR1km_standard.nc")
vars_needed = ['MWIR_Fire_pixel_BT', 'S8_Fire_pixel_BT', 'FRP_MWIR', 'TCWV', 'solar_zenith', 'sat_zenith', 'confidence_MWIR']
print(vars_needed)
ds_small = ds[vars_needed]
df = ds_small.to_dataframe().reset_index()

X = df[['MWIR_Fire_pixel_BT', 'S8_Fire_pixel_BT', 'FRP_MWIR', 'TCWV', 'solar_zenith', 'sat_zenith']]
print(X)

df['label'] = (df['confidence_MWIR'] > 80).astype(int)
y = df['label']
print(df['confidence_MWIR'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Acuratețe:", metrics.accuracy_score(y_test, y_pred))


joblib.dump(model, 'fire_detection_tree.pkl')

plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=['non-fire', 'fire'], filled=True)
plt.show()


rules = export_text(model, feature_names=list(X.columns))
print(rules)


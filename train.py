import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("mitbih_dataset.csv")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

# =========================
# FIND LABEL COLUMN
# =========================
label_col = None
for col in df.columns:
    if "type" in col or "label" in col or "class" in col:
        label_col = col
        break

id_col = None
for col in df.columns:
    if "record" in col or "patient" in col:
        id_col = col
        break

drop_cols = [label_col]
if id_col:
    drop_cols.append(id_col)

X = df.drop(columns=drop_cols)
y = df[label_col]

# =========================
# ENCODE LABELS
# =========================
le = LabelEncoder()
y = le.fit_transform(y)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL (BEST ONE ONLY FOR DEPLOYMENT)
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
pred = model.predict(X_test)

print("\nF1 Score:", f1_score(y_test, pred, average='macro'))
print(classification_report(y_test, pred))

# =========================
# SAVE EVERYTHING (IMPORTANT FOR STREAMLIT)
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved successfully!")
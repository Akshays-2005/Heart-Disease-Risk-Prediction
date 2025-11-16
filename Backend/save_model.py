import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

DATA_PATH = "../heart.csv"  
ARTIFACT_DIR = "artifacts"

def build_and_save():
    df = pd.read_csv(DATA_PATH)
    print("Loaded", DATA_PATH, "shape:", df.shape)

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Saved encoder for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop("target", axis=1)
    y = df["target"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Trained RandomForest. Accuracy on test set:", acc)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(ARTIFACT_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(ARTIFACT_DIR, "imputer.joblib"))
    joblib.dump(label_encoders, os.path.join(ARTIFACT_DIR, "encoders.joblib"))

    print("Saved artifacts to", ARTIFACT_DIR)

if __name__ == "__main__":
    build_and_save()

# heart_model_full_testcases_ensemble.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1️⃣ Load the dataset
# -----------------------------
df = pd.read_csv("heart.csv")
print("Dataset loaded. First 5 rows:\n", df.head())

# -----------------------------
# 2️⃣ Encode categorical columns
# -----------------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"{col} mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# -----------------------------
# 3️⃣ Separate features and target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# 4️⃣ Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# 5️⃣ Feature scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Preprocessing complete. Feature shape:", X.shape, "Target shape:", y.shape)

# -----------------------------
# 6️⃣ Split data into training and testing sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split complete. X_train:", X_train.shape, "X_test:", X_test.shape)

# -----------------------------
# 7️⃣ Train models
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("All models trained successfully.")

# -----------------------------
# 8️⃣ Evaluate models
# -----------------------------
models = {
    "Random Forest": rf,
    "Logistic Regression": lr,
    "KNN": knn
}

accuracies = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

best_model_name = max(accuracies, key=accuracies.get)
print(f"\nBest performing model: {best_model_name} with accuracy {accuracies[best_model_name]:.4f}")

# -----------------------------
# 9️⃣ Test cases for new patients
# -----------------------------
print("\nRunning predefined test cases...\n")

# Define test cases: strings for categorical features
test_cases = [
    {
        "age": 45,
        "sex": "Female",
        "chest_pain_type": "Typical angina",
        "resting_blood_pressure": 120,
        "cholestoral": 200,
        "fasting_blood_sugar": "Lower than 120 mg/ml",
        "rest_ecg": "Normal",
        "thalach": 160,
        "exercise_induced_angina": "No",
        "oldpeak": 1.0,
        "slope": "Flat",
        "vessels_colored_by_flourosopy": "Zero",
        "thalassemia": "Reversable Defect"
    },
    {
        "age": 63,
        "sex": "Male",
        "chest_pain_type": "Asymptomatic",
        "resting_blood_pressure": 145,
        "cholestoral": 233,
        "fasting_blood_sugar": "Greater than 120 mg/ml",
        "rest_ecg": "Left ventricular hypertrophy",
        "thalach": 150,
        "exercise_induced_angina": "Yes",
        "oldpeak": 2.3,
        "slope": "Upsloping",
        "vessels_colored_by_flourosopy": "Two",
        "thalassemia": "Reversable Defect"
    }
]

# Feature order (match CSV columns exactly)
feature_order = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholestoral",
    "fasting_blood_sugar",
    "rest_ecg",
    "thalach",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "vessels_colored_by_flourosopy",
    "thalassemia"
]

# Predict for each test case
for i, case in enumerate(test_cases, start=1):
    # Encode categorical features
    patient_values = []
    for feature in feature_order:
        if feature in label_encoders:
            patient_values.append(label_encoders[feature].transform([case[feature]])[0])
        else:
            patient_values.append(case[feature])
    
    # Scale
    patient_scaled = scaler.transform([patient_values])
    
    print(f"\nTest Case {i}:")
    ensemble_votes = []
    
    for name, model in models.items():
        pred_class = model.predict(patient_scaled)[0]
        pred_prob = model.predict_proba(patient_scaled)[0][1]  # probability of '1' (heart disease)
        ensemble_votes.append(pred_class)
        print(f"{name} Prediction: {'Yes' if pred_class==1 else 'No'} (Prob: {pred_prob:.2f})")
    
    # Ensemble prediction (majority vote)
    final_vote = "Yes" if sum(ensemble_votes) >= 2 else "No"
    print(f"Ensemble Prediction: {final_vote}")

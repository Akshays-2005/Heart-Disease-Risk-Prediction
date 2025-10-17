import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")
print("Dataset loaded. First 5 rows:\n", df.head())

label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"{col} mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df.drop("target", axis=1)
y = df["target"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Preprocessing complete. Feature shape:", X.shape, "Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split complete. X_train:", X_train.shape, "X_test:", X_test.shape)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("All models trained successfully.")

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

print("\nEnter new patient details for heart disease prediction:\n")

print("Use the following numeric mappings for categorical inputs:\n")

for col, le in label_encoders.items():
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"{col}: {mapping}")
print("\n")

patient_data = {}
for feature in [
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
]:
    value = float(input(f"Enter {feature.replace('_', ' ')}: "))
    patient_data[feature] = value

patient_values = [patient_data[f] for f in [
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
]]

patient_scaled = scaler.transform([patient_values])

print("\nPrediction Results:\n")
ensemble_votes = []

for name, model in models.items():
    pred_class = model.predict(patient_scaled)[0]
    pred_prob = model.predict_proba(patient_scaled)[0][1]
    ensemble_votes.append(pred_class)
    print(f"{name} Prediction: {'Yes' if pred_class == 1 else 'No'} (Prob: {pred_prob:.2f})")

final_vote = "Yes" if sum(ensemble_votes) >= 2 else "No"
print(f"\n✅ Final Ensemble Prediction: {final_vote}")

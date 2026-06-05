# ============================================================
# Smart Stethoscope AI - Random Forest Training
# Dataset: data/heart_features_jkall.csv
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
csv_file = "data/heart_features_jkall.csv"

df = pd.read_csv(csv_file)

print("====================================")
print("Dataset Loaded Successfully")
print("====================================")
print("File used:", csv_file)
print("Dataset shape:", df.shape)
print()

print("Columns:")
print(df.columns)
print()

print("First 5 rows:")
print(df.head())
print()


# 2. Check labels
print("====================================")
print("Label Count")
print("====================================")
print(df["label"].value_counts())
print()


# 3. Check missing values
print("====================================")
print("Missing Values")
print("====================================")
print(df.isnull().sum())
print()


# 4. Prepare features and label
X = df.drop(columns=["label", "file"])
y = df["label"]


# 5. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("====================================")
print("Label Mapping")
print("====================================")

for label_name, label_number in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(label_name, "=", label_number)

print()


# 6. Split train/test by audio file
groups = df["file"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(splitter.split(X, y_encoded, groups))

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y_encoded[train_index]
y_test = y_encoded[test_index]

print("====================================")
print("Train/Test Split")
print("====================================")
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Training files:", df.iloc[train_index]["file"].nunique())
print("Testing files:", df.iloc[test_index]["file"].nunique())
print()


# 7. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("====================================")
print("Model Training Completed")
print("====================================")
print()


# 8. Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("====================================")
print("Model Accuracy")
print("====================================")
print("Accuracy:", accuracy)
print()

print("====================================")
print("Classification Report")
print("====================================")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))
print()

print("====================================")
print("Confusion Matrix")
print("====================================")
print(confusion_matrix(y_test, y_pred))
print()


# 9. Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("====================================")
print("Top Important Features")
print("====================================")
print(feature_importance.head(10))
print()

feature_importance.to_csv("jkall_feature_importance.csv", index=False)


# 10. Save model files
joblib.dump(model, "heart_random_forest_model_jkall.pkl")
joblib.dump(label_encoder, "label_encoder_jkall.pkl")

print("====================================")
print("Files Saved")
print("====================================")
print("Model saved as: heart_random_forest_model_jkall.pkl")
print("Label encoder saved as: label_encoder_jkall.pkl")
print("Feature importance saved as: jkall_feature_importance.csv")
print()

print("Training finished successfully.")
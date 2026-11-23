import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.prediction import FEATURES

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_dataset.csv"
MODELS_DIR = BASE_DIR / "models"


def build_pipeline(model):
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("numerical", numerical, FEATURES)])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_models(data_path=DATA_PATH):
    dataframe = pd.read_csv(data_path)
    required = set(FEATURES + ["Final_Marks", "Performance_Category"])
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dataframe = dataframe.dropna(subset=FEATURES + ["Final_Marks", "Performance_Category"])
    X = dataframe[FEATURES]
    y_final = dataframe["Final_Marks"]
    y_category = dataframe["Performance_Category"]

    X_train, X_test, y_final_train, y_final_test, y_category_train, y_category_test = train_test_split(
        X, y_final, y_category, test_size=0.2, random_state=42
    )

    regressor = build_pipeline(RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2, max_features="sqrt", n_jobs=-1
    ))
    regressor.fit(X_train, y_final_train)
    final_predictions = regressor.predict(X_test)

    classifier = build_pipeline(RandomForestClassifier(
        n_estimators=300, random_state=42, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1
    ))
    classifier.fit(X_train, y_category_train)
    category_predictions = classifier.predict(X_test)

    labels = ["Excellent", "Good", "Average", "Needs Improvement"]
    metrics = {
        "mae": float(mean_absolute_error(y_final_test, final_predictions)),
        "rmse": float(root_mean_squared_error(y_final_test, final_predictions)),
        "r2_score": float(r2_score(y_final_test, final_predictions)),
        "accuracy": float(accuracy_score(y_category_test, category_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_category_test, category_predictions)),
        "classification_report": classification_report(y_category_test, category_predictions, labels=labels, output_dict=True, zero_division=0),
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y_category_test, category_predictions, labels=labels).tolist(),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "regressor_feature_importance": dict(zip(FEATURES, [float(value) for value in regressor.named_steps["model"].feature_importances_])),
        "classifier_feature_importance": dict(zip(FEATURES, [float(value) for value in classifier.named_steps["model"].feature_importances_])),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(regressor, MODELS_DIR / "performance_regressor.pkl")
    joblib.dump(classifier, MODELS_DIR / "performance_classifier.pkl")
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Regression MAE:", round(metrics["mae"], 2))
    print("Regression RMSE:", round(metrics["rmse"], 2))
    print("Regression R²:", round(metrics["r2_score"], 3))
    print("Classification Accuracy:", round(metrics["accuracy"] * 100, 2), "%")


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run generate_dataset.py first.")
    train_models()

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "spotify_preprocessing", "spotify_clean.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Kolom berikut tidak ditemukan pada dataset: {missing_columns}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    print("Dataset berhasil dimuat.")
    print("Shape dataset:", df.shape)
    print("\nDistribusi label:")
    print(df["label"].value_counts())

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("smote", SMOTE(
            random_state=42,
            k_neighbors=5
        )),
        ("svm", SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=42
        ))
    ])

    mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if mlflow_run_id:
        run_context = mlflow.start_run(run_id=mlflow_run_id)
    else:
        mlflow.set_experiment("Spotify Sentiment Classification TFIDF SMOTE SVM")
        run_context = mlflow.start_run(run_name="tfidf_smote_svm_baseline")

    with run_context:
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print("\nTraining selesai.")
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1-score :", f1)

        mlflow.log_param("feature_extraction", "TF-IDF")
        mlflow.log_param("resampling", "SMOTE")
        mlflow.log_param("model_type", "SVM")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", "(1, 2)")
        mlflow.log_param("tfidf_min_df", 2)
        mlflow.log_param("smote_random_state", 42)
        mlflow.log_param("smote_k_neighbors", 5)
        mlflow.log_param("svm_kernel", "linear")
        mlflow.log_param("svm_C", 1.0)
        mlflow.log_param("svm_probability", True)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        report = classification_report(y_test, y_pred)
        report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.txt")

        with open(cm_path, "w", encoding="utf-8") as file:
            file.write(str(cm))

        model_path = os.path.join(ARTIFACT_DIR, "spotify_tfidf_smote_svm_model.pkl")
        joblib.dump(model_pipeline, model_path)

        mlflow.sklearn.log_model(model_pipeline, "model")
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(model_path)

        print("\nArtefak berhasil disimpan:")
        print(report_path)
        print(cm_path)
        print(model_path)


if __name__ == "__main__":
    main()
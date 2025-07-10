import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import sys
import os
from io import StringIO
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
def clean_data(df):
    df = df.replace("?", pd.NA)
    df = df.dropna(axis=1, how='all')
    df = df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'name' in col.lower()])
    df = df.dropna()
    return df

def encode_features(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, label_encoders

def model_training(df, buffer, original_stdout, label_encoders):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ➤ Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

    # Start MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("DecisionTreeExperiment")
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="decisiontree_cv10"):
        # ➤ Train model
        clf = DecisionTreeClassifier(criterion="entropy", random_state=70, ccp_alpha=0.03)
        mlflow.log_param("criterion", "entropy")
        mlflow.log_param("random_state", 70)
        mlflow.log_param("ccp_alpha", 0.03)

        print("\n🔁 10-Fold Cross-Validation Scores:")
        scores = cross_validate(clf, X_train, y_train, cv=10)
        avg_accuracy = scores['test_score'].mean()
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        mlflow.log_metric("average_accuracy", avg_accuracy)

        clf.fit(X_train, y_train)

        # ➤ Feature importance
        features = pd.DataFrame({
            "Features": X.columns,
            "Importance": clf.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        print("\n💡 Feature Importances:")
        print(features)
        features.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_importance.csv"))

        # ➤ Validation evaluation
        y_val_pred = clf.predict(X_val)
        print("\n🔍 Validation Set Evaluation:")
        print(classification_report(y_val, y_val_pred))

        # ➤ Test evaluation
        y_test_pred = clf.predict(X_test)
        print("\n✅ Test Set Evaluation:")
        print(classification_report(y_test, y_test_pred))

        # ➤ Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print("\n🧮 Confusion Matrix (Test Set):\n")
        print(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Test Set)")
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # ➤ Decision tree plot
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_],
                  filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree (Trained on Training Set)")
        tree_path = os.path.join(OUTPUT_DIR, "decision_tree_sklearn.png")
        plt.savefig(tree_path)
        plt.close()
        mlflow.log_artifact(tree_path)

        # ➤ Write captured output
        result_path = os.path.join(OUTPUT_DIR, "result.txt")
        with open(result_path, "w") as f:
            f.write(buffer.getvalue())
        mlflow.log_artifact(result_path)

          # ➤ Log and register model (✅ corrected single call)
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="BeastCancer_DT",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            signature=infer_signature(X_val, clf.predict(X_val)),
            input_example=X_val.head(1)
        )

        print(f"✅ Model registered: {model_info.model_uri}")

        # ✅ Add this block to confirm registration
        client = MlflowClient()

        try:
            registered_model = client.get_registered_model("BeastCancer_DT")
            print(f"🔍 Registered model: {registered_model.name}")
            
            versions = client.search_model_versions(f"name='BeastCancer_DT'")
            for v in versions:
                print(f"📦 Version: {v.version}, Status: {v.status}, Source: {v.source}")
        except Exception as e:
            print(f"❌ Error fetching registered model info: {e}")

        sys.stdout = original_stdout
        print("✅ All results saved and logged to MLflow.")

def main():
    original_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    try:
        df = pd.read_csv(DATASET_PATH)
        df = clean_data(df)
        df, label_encoders = encode_features(df)
        model_training(df, buffer, original_stdout, label_encoders)
    finally:
        print("📁 Results saved to training.txt")

if __name__ == "__main__":
    main()

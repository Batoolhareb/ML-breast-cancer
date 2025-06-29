import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import os
from io import StringIO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

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



    # ➤ Train model
    clf = DecisionTreeClassifier(criterion="entropy", random_state=70, ccp_alpha=0.03)
    print("\n🔁 10-Fold Cross-Validation Scores:")
    scores = cross_validate(clf, X_train, y_train, cv=10)
    for key, value in scores.items():
        print(f"{key}: {value}")
    print(f"Average Accuracy: {scores['test_score'].mean():.4f}")
    
    clf.fit(X_train, y_train)

    # ➤ Feature importance
    features = pd.DataFrame({
        "Features": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n💡 Feature Importances:")
    print(features)
    features.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

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
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # ➤ Decision tree plot
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree (Trained on Training Set)")
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_sklearn.png"))
    plt.close()

    # ➤ Write captured output
    with open(os.path.join(OUTPUT_DIR, "result.txt"), "w") as f:
        f.write(buffer.getvalue())

    # ➤ Save model
    joblib.dump(clf, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.joblib"))

    sys.stdout = original_stdout
    print("✅ All results saved.")

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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
from io import StringIO

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
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ➤ Split: 80% training+validation, 20% testing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    # ➤ Split training+validation into 70% training, 30% validation
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

    print(f"\n📁 Data Split Summary:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # ➤ Train model
    clf = DecisionTreeClassifier(criterion="entropy", random_state=70, ccp_alpha=0.03)
    print("\n🔁 10-Fold Cross-Validation Scores:")
    scores = cross_validate(clf, X_train, y_train, cv=10)
    for key, value in scores.items():
        print(f"{key}: {value}")
    print(f"Average Accuracy: {scores['test_score'].mean():.4f}")
    
    clf.fit(X_train, y_train)

    # ➤ Feature Importance
    features = pd.DataFrame({
        "Features": X.columns,
        "Importance": clf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\n💡 Feature Importances:")
    print(features)
    features.to_csv("../output/feature_importance.csv", index=False)

    # ➤ Evaluate on validation set
    y_val_pred = clf.predict(X_val)
    print("\n🔍 Validation Set Evaluation:")
    print(classification_report(y_val, y_val_pred))

    # ➤ Evaluate on test set
    y_test_pred = clf.predict(X_test)
    print("\n✅ Test Set Evaluation:")
    print(classification_report(y_test, y_test_pred))

    # ➤ Confusion matrix for test
    cm = confusion_matrix(y_test, y_test_pred)
    print("\n🧮 Confusion Matrix (Test Set):\n")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig("../output/confusion_matrix.png")
    plt.close()

    # ➤ Save decision tree plot
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree (Trained on Training Set)")
    plt.savefig("../output/decision_tree_sklearn.png")
    plt.close()

    # ➤ Write captured output to file
    with open("../output/result.txt", "w") as f:
        f.write(buffer.getvalue())

    # ➤ Save model and encoders
    joblib.dump(clf, "../model/model.joblib")
    joblib.dump(label_encoders, "../model/encoders.joblib")
    joblib.dump(X.columns.tolist(), "../model/feature_names.joblib")

    sys.stdout = original_stdout
    print("✅ All results saved to:")
 
    
def main():
    original_stdout = sys.stdout
    sys.stdout = buffer = StringIO()

    try:
        df = pd.read_csv("../dataset/dataset.csv")
        df = clean_data(df)
        df, label_encoders = encode_features(df)
        model_training(df, buffer, original_stdout, label_encoders)

    finally:
        print("📁 Results saved to training.txt")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

    # Binary target: phishing vs legitimate
    target_col = "label"

    # Use only numeric columns as features for a simple scikit-learn baseline
    feature_cols = [
        c for c in df.select_dtypes(include=["number"]).columns.tolist()
        if c != target_col
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_imp, y_train)

    y_pred = model.predict(X_test_imp)

    F1_score = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", F1_score)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Show coefficients for the strongest positive/negative predictors
    coef = model.coef_[0]
    coef_rank = sorted(
        zip(feature_cols, coef),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]

    print("\nTop 10 feature weights:")
    for feature, weight in coef_rank:
        print(f"{feature}: {weight:.6f}")


if __name__ == "__main__":
    main()

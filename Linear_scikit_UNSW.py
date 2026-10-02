import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # Load dataset
    df = pd.read_csv("unsw-nb15/versions/1/UNSW_NB15_training-set.csv")

    # Target: predict total bytes sent
    target_col = "dur"  # Replace with the actual target column name if different

    # Keep only numeric features; exclude the target and ID columns
    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in {"id", target_col}]

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fill missing values and train linear regression
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    predictions = model.predict(X_test_imputed)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Model: Linear Regression")
    print("Target:", target_col)
    print("Features used:", len(feature_cols))
    print("RMSE:", rmse)
    print("R^2:", r2)

if __name__ == "__main__":
    main()

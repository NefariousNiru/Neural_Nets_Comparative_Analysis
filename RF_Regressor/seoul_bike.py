from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datasets.seoul_bike_sharing_demand import seoul_bike
from util import pre_processing
from validation import in_sample_validation, cross_validation, train_test_split_validation

def run_seoul_bike():
    data = seoul_bike.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    # Separate target and features
    X = data.drop('Rented Bike Count', axis=1)
    y = data['Rented Bike Count']

    # Model pipeline
    pipeline = Pipeline(steps=[
        ('regressor', RandomForestRegressor())
    ])

    # Hyperparameter tuning
    param_grid = {
        'regressor__n_estimators': [100, 400],
        'regressor__max_depth': [10, 30],
        'regressor__min_samples_leaf': [1, 2, 6]
    }

    in_sample_r2 = in_sample_validation(X, y, pipeline)
    print(f"In-sample R^2 score: {in_sample_r2:.4f}")
    test_r2 = train_test_split_validation(X, y, 0.2, 42, pipeline)
    print(f"Train-test R^2 score: {test_r2:.4f}")
    cross_val_r2 = cross_validation(X, y, 5, pipeline, param_grid)
    print(f"Cross-validation R^2 score: {cross_val_r2:.4f}")

if __name__ == "__main__":
    print("Random Forest Regression for Seoul Bike")
    run_seoul_bike()
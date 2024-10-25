from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datasets.auto_mpg import auto_mpg
from util import pre_processing
from validation import in_sample_validation, cross_validation, train_test_split_validation

def run_auto_mpg():
    data = auto_mpg.get_dataset()
    data = pre_processing.drop_outliers_inter_quartile(data)
    print(f"\nShape after removing Outliers using IQR, threshold 1.5, {data.shape}\n")

    # Separate target and features
    X = data.drop('mpg', axis=1)
    y = data['mpg']

    # Define categorical and numerical features
    categorical_features = ['origin']
    numerical_features = [col for col in X.columns if col != 'origin']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    # Model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Hyperparameter tuning
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 25],
        'regressor__min_samples_leaf': [1, 2, 5]
    }

    in_sample_r2 = in_sample_validation(X, y, pipeline)
    test_r2 = train_test_split_validation(X, y, 0.2, 42, pipeline)
    cross_val_r2 = cross_validation(X, y, 5, pipeline, param_grid)

    print(f"In-sample R^2 score: {in_sample_r2:.4f}")
    print(f"Train-test R^2 score: {test_r2:.4f}")
    print(f"Cross-validation R^2 score: {cross_val_r2:.4f}")

if __name__ == "__main__":
    print("Random Forest Regression for Auto MPG")
    run_auto_mpg()
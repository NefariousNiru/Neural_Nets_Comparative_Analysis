import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from Custom_NN.CustomNN import CustomNN
from feature_selection.train_test import train_and_evaluate
from util import performance_metrics

def backward_feature_selection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    selected_features = list(X.columns)

    # Calculate initial R² with all features
    initial_r2 = train_and_evaluate(X_train[selected_features], y_train, X_test[selected_features], y_test, len(selected_features))
    print(f"Initial R² with all features: {initial_r2:.4f}")

    # Iteratively remove features and check model performance
    best_r2 = initial_r2
    for feature in selected_features[:]:  # Work on a copy of features list
        trial_features = [f for f in selected_features if f != feature]
        r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))

        print(f"Feature removed: {feature}, R² Score: {r2_score:.4f}")

        # Remove feature if R² improves or stays the same
        if r2_score >= best_r2:
            best_r2 = r2_score
            selected_features.remove(feature)
            print(f"Removed {feature}, updated best R²: {best_r2:.4f}")
        else:
            print(f"Kept {feature}, R² did not improve.")

    print(f"Final selected features: {selected_features}")
    print(f"Final selected features R2: {best_r2}")
    return selected_features



def backward_feature_selection_categorical(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize with all features selected
    selected_features = list(X.columns)

    # Group categorical features by prefix
    grouped_features = {}
    for feature in X.columns:
        if feature.startswith("Seasons"):
            grouped_features.setdefault("Seasons", []).append(feature)
        elif feature.startswith("Holiday"):
            grouped_features.setdefault("Holiday", []).append(feature)
        else:
            grouped_features[feature] = [feature]  # Treat numeric features as single-member groups

    # Calculate initial R² with all features
    initial_r2 = train_and_evaluate(X_train[selected_features], y_train, X_test[selected_features], y_test, len(selected_features))
    print(f"Initial R² with all features: {initial_r2:.4f}")

    best_r2 = initial_r2
    for group, features in grouped_features.items():
        # Try removing the entire group
        trial_features = [f for f in selected_features if f not in features]
        r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))

        print(f"Group removed: {group}, R² Score: {r2_score:.4f}")

        # Remove the group if R² improves or stays the same
        if r2_score >= best_r2:
            best_r2 = r2_score
            selected_features = trial_features
            print(f"Removed group: {group}, updated best R²: {best_r2:.4f}")
        else:
            print(f"Kept group: {group}, R² did not improve.")

    print(f"Final selected features: {selected_features}")
    print(f"Final selected features R²: {best_r2:.4f}")
    return selected_features
from sklearn.model_selection import train_test_split
from feature_selection.train_test import train_and_evaluate

def forward_feature_selection(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize with no features
    selected_features = []
    remaining_features = list(X.columns)

    # Initialize best R² to a very low value
    best_r2 = float('-inf')

    for feature in remaining_features:
        trial_features = selected_features + [feature]

        r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))
        print(f"Evaluating feature: {feature}, R² Score: {r2_score:.4f}")

        # If the best R² improves, add the feature
        if r2_score > best_r2:
            best_r2 = r2_score
            selected_features.append(feature)
            print(f"Added feature: {feature}, updated best R²: {best_r2:.4f}")

    print(f"Final selected features: {selected_features}")
    print(f"Final selected features R²: {best_r2:.4f}")
    return selected_features


from sklearn.model_selection import train_test_split
from feature_selection.train_test import train_and_evaluate


def forward_feature_selection_categorical(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize with no features selected and set best R² to a very low value
    selected_features = []
    best_r2 = float('-inf')

    # Group categorical features by prefixes
    grouped_features = {}
    for feature in X.columns:
        if feature.startswith("Seasons"):
            grouped_features.setdefault("Seasons", []).append(feature)
        elif feature.startswith("Holiday"):
            grouped_features.setdefault("Holiday", []).append(feature)
        else:
            grouped_features[feature] = [feature]  # Treat individual numeric features as single-member groups

    # Single loop over remaining groups/features
    for group, features in grouped_features.items():
        # Trial with the current feature group
        trial_features = selected_features + features
        r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))
        print(f"Evaluating group: {group}, R² Score: {r2_score:.4f}")

        # If adding the group improves R², include it in selected features
        if r2_score > best_r2:
            best_r2 = r2_score
            selected_features.extend(features)
            print(f"Added group: {group}, updated best R²: {best_r2:.4f}")

    print(f"Final selected features: {selected_features}")
    print(f"Final selected features R²: {best_r2:.4f}")
    return selected_features
from sklearn.model_selection import train_test_split
from feature_selection.train_test import train_and_evaluate

def stepwise_feature_selection(X, y):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    selected_features = []  # Start with no features selected
    remaining_features = list(X.columns)  # All features are initially available for selection
    best_r2 = float('-inf')  # Start with the lowest possible R² score

    while remaining_features:
        # Forward selection: add the best feature that improves R² the most
        best_feature = None
        best_trial_r2 = best_r2  # Track the best R² in this iteration

        for feature in remaining_features:
            trial_features = selected_features + [feature]
            r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))
            print(f"Evaluating feature: {feature}, R² Score: {r2_score:.4f}")

            # Check if this feature improves R²
            if r2_score > best_trial_r2:
                best_trial_r2 = r2_score
                best_feature = feature

        # Add the best feature only if it improves the overall R²
        if best_feature and best_trial_r2 > best_r2:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_r2 = best_trial_r2  # Update the best overall R²
            print(f"Added feature: {best_feature}, updated best R²: {best_r2:.4f}")
        else:
            # If no remaining feature improves R², stop
            print("No further features improved R²; stopping selection.")
            break

        # Backward elimination: try removing features to see if it improves R²
        for feature in selected_features[:]:  # Copy the list to iterate safely
            trial_features = [f for f in selected_features if f != feature]
            r2_score = train_and_evaluate(X_train[trial_features], y_train, X_test[trial_features], y_test, len(trial_features))
            print(f"Evaluating feature removal: {feature}, R² Score: {r2_score:.4f}")

            if r2_score >= best_r2:
                selected_features.remove(feature)
                best_r2 = r2_score
                print(f"Removed feature: {feature}, updated best R²: {best_r2:.4f}")

    print(f"Final selected features: {selected_features}")
    print(f"Final selected features R²: {best_r2:.4f}")
    return selected_features

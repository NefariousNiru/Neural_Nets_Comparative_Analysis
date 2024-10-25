from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

def in_sample_validation(X, y, pipeline):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    in_sample_r2 = r2_score(y, y_pred)
    return in_sample_r2

def train_test_split_validation(X, y, test_size, random_state, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_r2 = r2_score(y_test, y_test_pred)
    return y_test_r2

def cross_validation(X, y, cv, pipeline, param_grid):
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2')
    grid_search.fit(X, y)
    cross_val_r2 = grid_search.best_score_
    return cross_val_r2
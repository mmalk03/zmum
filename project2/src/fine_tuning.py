import datetime
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def _print_search_report(grid_search, n_top=3):
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best Balanced accuracy: {grid_search.best_score_}")

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(grid_search.cv_results_['rank_test_score'] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print("Mean validation Balanced accuracy: {0:.3f} (std: {1:.3f})".format(
                grid_search.cv_results_['mean_test_score'][candidate],
                grid_search.cv_results_['std_test_score'][candidate])
            )
            print(f"Parameters: {grid_search.cv_results_['params'][candidate]}")


def _fine_tune(model, params, x, y):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring='balanced_accuracy',
        refit='balanced_accuracy',
        cv=5,
        # n_jobs=-1,
        verbose=10
    )
    start_time = time.time()
    grid_search.fit(x, y)
    end_time = time.time()
    print(f"Grid search finished in: {str(datetime.timedelta(seconds=end_time - start_time))}")
    _print_search_report(grid_search)
    return grid_search.best_estimator_


def fine_tune_lightgbm(x, y):
    params = {
        'max_bin': [205, 255, 305],
        'num_leaves': [21, 26, 31, 36, 41],
        'n_estimators': [60, 80, 100, 120, 140],
    }
    model = LGBMClassifier(random_state=42)
    return _fine_tune(model, params, x, y)


def fine_tune_xgboost(x, y):
    kwargs = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    }
    params = {
        'max_depth': [3, 4, 5, 6],  # 5
        'n_estimators': [150, 200, 250],  # 200
        'gamma': [0, 0.2],  # 0
        'subsample': [0.8, 1.0],  # 0.8
        'colsample_bytree': [0.8, 1.0],  # 0.8
    }
    model = XGBClassifier(random_state=42, **kwargs)
    return _fine_tune(model, params, x, y)

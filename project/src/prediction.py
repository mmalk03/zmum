import datetime
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import data_processing as data_processing
import data_repository as repository
import plot_factory


def accuracy_at_10(y, y_pred_proba):
    df = pd.DataFrame(data={'actual': list(y), 'predicted': y_pred_proba})
    sorted_df = df.sort_values(by='predicted', ascending=False)
    top_ten_percent = sorted_df[:int(len(sorted_df) * 0.1)]
    score = round(sum(top_ten_percent['actual']) / len(top_ten_percent), 4)
    print(f"Accuracy@10: {score * 100}%")
    return score


def get_scores(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)[:, 1]
    return {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'AUC': roc_auc_score(y_val, y_pred_proba),
        'Accuracy@10': accuracy_at_10(y_val, y_pred_proba)
    }


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_Accuracy@10'] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print("Mean validation Accuracy@10: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_Accuracy@10'][candidate],
                results['std_test_Accuracy@10'][candidate])
            )
            print(f"Parameters: {results['params'][candidate]}")


def get_xgboost_important_features(model, columns):
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    df = pd.DataFrame(data={
        'Feature': [columns[index] for index in sorted_idx],
        'Importance': [model.feature_importances_[index] for index in sorted_idx]
    })
    return list(df['Feature'].head(n=30))


def get_lightgbm_important_features(model, columns):
    fig, ax = plt.subplots(figsize=(12, 18))
    lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    df = pd.DataFrame(data={
        'Feature': [columns[index] for index in sorted_idx],
        'Importance': [model.feature_importances_[index] for index in sorted_idx]
    })
    return list(df['Feature'].head(n=30))


def get_catboost_important_features(model):
    feature_importance = model.get_feature_importance(prettified=True)
    features, importances = list(zip(*feature_importance))
    df = pd.DataFrame(data={'Feature': features, 'Importance': importances})
    return list(df['Feature'].head(n=30))


def main():
    x, y, x_test = data_processing.prepare_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)
    neg_to_pos_ratio = round(len(y.loc[y == 0]) / len(y.loc[y == 1]))

    models = {
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
        'XGBoost balanced': XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42),
        'LightGBM balanced': LGBMClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        'CatBoost balanced': CatBoostClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42, verbose=0),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'BalancedRandomForest': BalancedRandomForestClassifier(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
        # 'MLP': MLPClassifier(random_state=42),
    }
    scores = {}
    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(x_train, y_train)
        end = time.time()
        print(f"Elapsed: {int((end - start) // 60)}m {int((end - start) % 60)}s")
        results.append((name, y_val, model.predict_proba(x_val)[:, 1]))
        scores[name] = get_scores(model, x_val, y_val)
    scores_df = pd.DataFrame(scores).transpose()
    plot_factory.plot_roc_curve(results)

    # Feature importance
    important_features = [
        get_xgboost_important_features(models[0], x.columns),
        get_xgboost_important_features(models[1], x.columns),
        get_lightgbm_important_features(models[2], x.columns),
        get_lightgbm_important_features(models[3], x.columns),
        get_catboost_important_features(models[4]),
        get_catboost_important_features(models[5]),
    ]
    unique_features = set([f for sublist in important_features for f in sublist])
    print(f"{len(unique_features)} unique features were chosen")

    x_train_small = x_train.filter(unique_features)
    x_val_small = x_val.filter(unique_features)
    scores_fi = {}
    results_fi = []
    for name, model in models.items():
        model.fit(x_train_small, y_train)
        results_fi.append((name, y_val, model.predict_proba(x_val)[:, 1]))
        scores_fi[name] = get_scores(model, x_val_small, y_val)
    scores_fi_df = pd.DataFrame(scores_fi).transpose()
    plot_factory.plot_roc_curve(results)

    # Fine tuning
    scoring = {'Accuracy@10': make_scorer(accuracy_at_10, needs_proba=True)}
    kwargs = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
    }
    params = {
        'scale_pos_weight': [1, 7, 13],  # 13
        'max_depth': [3, 4, 5],  # 5
        'min_child_weight': [7],  # 7
        'learning_rate': [0.01],  # 0.01
        'n_estimators': [150, 200],  # 200
        'gamma': [0, 0.2],  # 0
        'subsample': [0.8, 1.0],  # 0.8
        'colsample_bytree': [0.8, 1.0],  # 0.8
    }
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42, **kwargs),
        param_grid=params,
        scoring=scoring,
        refit='Accuracy@10',
        cv=3,
        # n_jobs=-1,
        verbose=10
    )
    start_time = time.time()
    grid_search.fit(x, y)
    end_time = time.time()
    print(f"Grid search finished in: {str(datetime.timedelta(seconds=end_time - start_time))}")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best Accuracy@10: {grid_search.best_score_}")
    report_best_scores(grid_search.cv_results_, 1)
    # best_params = []
    # best_score = []
    best_params += [grid_search.best_params_]
    best_score += [grid_search.best_score_]

    best_model = grid_search.best_estimator_
    # best_model.fit(x_train, y_train)
    scores_train = get_scores(best_model, x_train, y_train)
    scores_val = get_scores(best_model, x_val, y_val)
    fine_tune_df = pd.DataFrame(data={
        f"XGBoost search 1 - train": scores_train,
        f"XGBoost search 1 - val": scores_val
    }).transpose()

    # Cross validation of the best model
    model = XGBClassifier(
        scale_pos_weight=13,
        max_depth=5,
        min_child_weight=7,
        learning_rate=0.01,
        n_estimators=200,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    scores_cv = {}
    results_cv = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_, (train_index, test_index) in enumerate(kfold.split(x)):
        x_t, x_v = x.iloc[train_index], x.iloc[test_index]
        y_t, y_v = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_t, y_t)
        results_cv.append((f"Fold {fold_}", y_val, model.predict_proba(x_val)[:, 1]))
        scores_cv[f"Fold {fold_}"] = get_scores(model, x_v, y_v)
    scores_cv_df = pd.DataFrame(scores_cv).transpose()
    plot_factory.plot_roc_curve(results_cv)

    # Final predictions
    best_model.fit(x, y)
    y_test_predictions = best_model.predict_proba(x_test)
    repository.save_results(list(y_test_predictions[:, 1]), best_model.__class__.__name__)


if __name__ == '__main__':
    main()

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from xgboost import XGBClassifier

import data_processing as data_processing
import data_repository as repository
import plot_factory


def evaluate(model, x_test, y_test):
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    df = pd.DataFrame(data={'actual': list(y_test), 'predicted': y_pred_proba})
    sorted_df = df.sort_values(by='predicted', ascending=False)
    top_ten_percent = sorted_df[:int(len(sorted_df) * 0.1)]
    score = round(sum(top_ten_percent['actual']) / len(top_ten_percent), 4)
    print(f"Accuracy@10 of {model.__class__.__name__} is: {score * 100}%")
    return score


def get_scores(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)[:, 1]

    scores = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'AUC': roc_auc_score(y_val, y_pred_proba),
        'Accuracy@10': evaluate(model, x_val, y_val)
    }

    # plot_factory.plot_roc_curve(y_val, y_pred_proba[:, 1])

    # report = classification_report(y, y_pred)
    # print(report)

    # y_val_pred_sorted = list(reversed(sorted(y_val_pred_proba[:, 1])))
    # chosen_results = y_val_pred_sorted[:round(0.1 * len(y_val_pred_sorted))]
    # print(f"10% of predicted probabilities are greater than {chosen_results[-1]}")

    return scores


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


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
        'XGBoost': XGBClassifier(random_state=42),
        'XGBoost balanced': XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'LightGBM balanced': LGBMClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42),
        'CatBoost balanced': CatBoostClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'BalancedRandomForest': BalancedRandomForestClassifier(random_state=42),
    }
    scores = {}
    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
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

    # Cross validation
    models_cv = {
        'XGBoost balanced': XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
    }
    scores_cv = {}
    results_cv = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models_cv.items():
        for fold_, (train_index, test_index) in enumerate(kfold.split(x)):
            x_t, x_v = x.iloc[train_index], x.iloc[test_index]
            y_t, y_v = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_t, y_t)
            results_cv.append((f"{name}-{fold_}", y_val, model.predict_proba(x_val)[:, 1]))
            scores_cv[f"{name}-{fold_}"] = get_scores(model, x_v, y_v)
    scores_cv_df = pd.DataFrame(scores_cv).transpose()
    plot_factory.plot_roc_curve(results_cv)

    # TODO do some GridSearch for XGBoost and submit the results

    # Hyperparameter searching
    model = XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42)
    # parameters = {
    #     'min_child_weight': [1, 2, 3, 4, 5],
    #     'reg_lambda': [0.50, 0.75, 1, 1.25, 1.5]
    # }
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        random_state=42,
        n_iter=200,
        cv=3,
        verbose=1,
        n_jobs=1,
        return_train_score=True)
    search.fit(x_train, y_train)
    report_best_scores(search.cv_results_, 1)

    # Final model
    # TODO use XGB after hyperparameter tuning
    best_model = XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42)
    best_model.fit(x, y)
    y_test_predictions = best_model.predict_proba(x_test)
    repository.save_results(list(y_test_predictions[:, 1]), best_model.__class__.__name__)


if __name__ == '__main__':
    main()

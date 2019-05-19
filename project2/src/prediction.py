import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import data_repository as repository
import feature_selection
import fine_tuning
import plot_factory


def get_scores(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)[:, 1]
    return {
        'Balanced accuracy': balanced_accuracy_score(y_val, y_pred),
        'Accuracy': accuracy_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'F1': f1_score(y_val, y_pred),
        'AUC': roc_auc_score(y_val, y_pred_proba)
    }


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


def main():
    x, y = repository.load_train_dataset()
    x_test = repository.load_test_dataset()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

    # Baselines
    models = {
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(random_state=42, n_jobs=-1)
    }
    scores = {}
    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        results.append((name, y_val, model.predict_proba(x_val)[:, 1]))
        scores[name] = get_scores(model, x_val, y_val)
    scores_df = pd.DataFrame(scores).transpose()
    plot_factory.plot_roc_curve(results)

    # Correlation matrix
    feature_selection.show_correlation_matrix_with_heatmap(x, y)

    # Univariate Selection
    features = feature_selection.select_features_by_univariate_selection(x_train, y_train)
    features.nlargest(20, 'Score').plot(kind='barh', x='Feature', y='Score')
    plt.show()

    # Feature importance
    important_features = [
        get_xgboost_important_features(models['XGBoost'], x.columns),
        get_lightgbm_important_features(models['LightGBM'], x.columns)
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

    # Recursive Feature Elimination
    rfe_xgb_model = XGBClassifier(random_state=42, n_jobs=-1)
    rfe_xgb_features = feature_selection.select_features_by_recursive_feature_elimination(rfe_xgb_model, x, y)

    # Recursive Feature Elimination with Cross Validation and Fine tuning

    # LightGBM
    rfe_cv_lgbm_model = LGBMClassifier(random_state=42)
    rfe_cv_lgbm_features = feature_selection.select_features_by_rfe_with_cv(rfe_cv_lgbm_model, x, y)
    # rfe_cv_lgbm_features = [106, 154, 319, 337, 379, 443, 454]
    x_rfe_cv_lgbm = x.filter(rfe_cv_lgbm_features)
    best_lgbm = fine_tuning.fine_tune_lightgbm(x_rfe_cv_lgbm, y)
    # Best params: {'max_bin': 255, 'n_estimators': 80, 'num_leaves': 41}
    # Best Balanced accuracy: 0.872
    x_test_rfe_cv_lgbm = x_test.filter(rfe_cv_lgbm_features)
    y_test_predictions_lgbm = best_lgbm.predict_proba(x_test_rfe_cv_lgbm)
    repository.save_results(
        model_name=best_lgbm.__class__.__name__,
        predictions=list(y_test_predictions_lgbm[:, 1]),
        features=rfe_cv_lgbm_features
    )

    # XGBoost
    rfe_cv_xgb_model = XGBClassifier(random_state=42)
    rfe_cv_xgb_features = feature_selection.select_features_by_rfe_with_cv(rfe_cv_xgb_model, x, y)
    # rfe_cv_xgb_features = [29, 49, 106, 154, 242, 282, 319, 339, 379, 443, 452, 473, 476]
    x_rfe_cv_xgb = x.filter(rfe_cv_xgb_features)
    best_xgb = fine_tuning.fine_tune_xgboost(x_rfe_cv_xgb, y)
    # Best params: {'colsample_bytree': 0.8, 'gamma': 0, 'max_depth': 5, 'n_estimators': 250, 'subsample': 0.8}
    # Best Balanced accuracy: 0.859
    x_test_rfe_cv_xgb = x_test.filter(rfe_cv_xgb_features)
    y_test_predictions_xgb = best_xgb.predict_proba(x_test_rfe_cv_xgb)
    repository.save_results(
        model_name=best_xgb.__class__.__name__,
        predictions=list(y_test_predictions_xgb[:, 1]),
        features=rfe_cv_xgb_features
    )


if __name__ == '__main__':
    main()

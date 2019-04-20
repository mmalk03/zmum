import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import get_dummies
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from xgboost import XGBClassifier

import data_processing as data_processing
import data_repository as repository
import plot_factory as plot_factory


def fit_and_save_scores(model, x_train, y_train, x_val, y_val):
    print(f"Fitting: {model.__class__.__name__}")
    scores = {}

    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred_proba = model.predict_proba(x_val)

    scores['Accuracy'] = accuracy_score(y_val, y_pred)
    scores['Recall'] = recall_score(y_val, y_pred)
    scores['Precision'] = precision_score(y_val, y_pred)
    scores['F1'] = f1_score(y_val, y_pred)
    scores['AUC'] = roc_auc_score(y_val, y_pred_proba[:, 1])

    # plot_factory.plot_roc_curve(y_val, y_pred_proba[:, 1])

    # report = classification_report(y, y_pred)
    # print(report)

    # y_val_pred_sorted = list(reversed(sorted(y_val_pred_proba[:, 1])))
    # chosen_results = y_val_pred_sorted[:round(0.1 * len(y_val_pred_sorted))]
    # print(f"10% of predicted probabilities are greater than {chosen_results[-1]}")

    return scores


def main():
    x, y, x_test = data_processing.prepare_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=False)

    neg_to_pos_ratio = round(len(y.loc[y == 0]) / len(y.loc[y == 1]))

    scores = {}
    models = [
        XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42),
        RandomForestClassifier(random_state=42),
        AdaBoostClassifier(random_state=42)
    ]
    for model in models:
        model_name = model.__class__.__name__
        scores[model_name] = {}
        scores[model_name] = fit_and_save_scores(model, x_train, y_train, x_val, y_val)
    scores_df = pd.DataFrame(scores).transpose()

    i = 0
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(x_train):
        x_t, x_v = x_train[train_index], x_train[test_index]
        y_t, y_v = y_train[train_index], y_train[test_index]

        model = XGBClassifier(scale_pos_weight=neg_to_pos_ratio, random_state=42)
        model_name = f"{model.__class__.__name__}-{i}"
        scores[model_name] = {}
        scores[model_name] = fit_and_save_scores(model, x_t, y_t, x_v, y_v)
        i += 1
    scores_df = pd.DataFrame(scores).transpose()


    best_model = models[0]
    y_test_predictions = best_model.predict_proba(x_test)
    repository.save_results(list(y_test_predictions[:, 1]), 'XGBoost')

    clf_tune = XGBClassifier(n_estimators=300)
    parameters = {'max_depth': range(5),
                  'min_child_weight': [1, 2, 3, 4, 5],
                  'reg_lambda': [0.50, 0.75, 1, 1.25, 1.5]}

    # LightGBM

    features = x_train.columns.values[:-1]
    target = x_train['class']
    params = {
        # 'num_leaves': 6,
        # 'max_bin': 63,
        # 'min_data_in_leaf': 45,
        # 'learning_rate': 0.01,
        # 'min_sum_hessian_in_leaf': 0.000446,
        # 'bagging_fraction': 0.55,
        # 'bagging_freq': 5,
        # 'max_depth': 14,
        'save_binary': True,
        'seed': 31452,
        'feature_fraction_seed': 31415,
        # 'feature_fraction': 0.51,
        'bagging_seed': 31415,
        'drop_seed': 31415,
        'data_random_seed': 31415,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
    folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=31415)
    oof = np.zeros(len(x_train))
    predictions = np.zeros(len(x_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (train_idx, valid_idx) in enumerate(folds.split(x_train.values, target.values)):
        print("Fold {}".format(fold_))
        train_data = lgb.Dataset(x_train.iloc[train_idx][features], label=target.iloc[train_idx])
        valid_data = lgb.Dataset(x_train.iloc[valid_idx][features], label=target.iloc[valid_idx])

        num_round = 15000
        clf = lgb.train(params, train_data, num_round,
                        valid_sets=[train_data, valid_data],
                        verbose_eval=1000,
                        early_stopping_rounds=250)
        oof[valid_idx] = clf.predict(x_train.iloc[valid_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df['Feature'] = features
        fold_importance_df['Importance'] = clf.feature_importance()
        fold_importance_df['Fold'] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(x_test[features], num_iteration=clf.best_iteration) / folds.n_splits
    # TODO check how this model works for rows which should be classified as 1
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    # Lets check the feature importance
    cols = (feature_importance_df[['Feature', 'Importance']]
            .groupby('Feature')
            .mean()
            .sort_values(by='Importance', ascending=False)[:150]
            .index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
    plt.figure(figsize=(14, 28))
    sns.barplot(x='Importance', y='Feature', data=best_features.sort_values(by='Importance', ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.show()

    print(f"{len(predictions[predictions > 0.5])}/{len(predictions)} rows were predicted as 1")
    # TODO looks nice on training set but not so nice on test set
    test_predictions = clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits
    print(f"{len(test_predictions[test_predictions > 0.5])}/{len(test_predictions)} rows were predicted as 1")

    # Random Forest and Gradient Boosting
    x_train = x_train.drop('class', axis=1)
    y_train = x_train['class']

    rf_model = RandomForestClassifier(
        n_estimators=1200,
        criterion='entropy',
        max_depth=None,  # expand until all leaves are pure or contain < MIN_SAMPLES_SPLIT samples
        min_samples_split=100,
        min_samples_leaf=50,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        # number of features to consider when looking for the best split; None: max_features=n_features
        max_leaf_nodes=None,  # None: unlimited number of leaf nodes
        bootstrap=True,
        oob_score=True,  # estimate Out-of-Bag Cross Entropy
        class_weight=None,  # our classes are skewed, but but too skewed
        random_state=123,
        verbose=0,
        warm_start=False)
    rf_model.fit(X=pd.get_dummies(x_train), y=y_train)

    boost_model = GradientBoostingClassifier(
        n_estimators=2400,
        loss='deviance',  # a.k.a Cross Entropy in Classification
        learning_rate=.01,  # shrinkage parameter
        subsample=1.,
        min_samples_split=200,
        min_samples_leaf=100,
        min_weight_fraction_leaf=0.0,
        max_depth=10,  # maximum tree depth / number of levels of interaction
        init=None,
        random_state=123,
        max_features=None,
        # number of features to consider when looking for the best split; None: max_features=n_features
        verbose=0,
        max_leaf_nodes=None,  # None: unlimited number of leaf nodes
        warm_start=False)
    boost_model.fit(X=get_dummies(x_train), y=y_train)

    # TODO firstly concatenate train and test dfs, convert them to dummies, and then split so that they have the same columns
    rf_train_pred_probs = rf_model.predict_proba(X=get_dummies(x_train))
    rf_test_pred_probs = rf_model.predict_proba(X=get_dummies(x_test))

    x = x_train
    for feature in x.columns:
        if str(x[feature].dtype) == 'category':
            x.pop(feature)
    y = x.pop('class')
    # model = LinearDiscriminantAnalysis()
    model = LogisticRegression()
    model.fit(x, y)
    # kfold = KFold(n_splits=3, random_state=7)
    # result = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    # print(result.mean())

    x['class'] = y
    new_x = x.loc[x['class'] == 1]
    new_x = x.loc[x['class'] == 0]
    new_x = x
    predictions = model.predict(new_x)

    sum(predictions) / len(predictions)

    from sklearn import metrics

    cm = metrics.confusion_matrix(new_x['class'], predictions)
    print(cm)


if __name__ == '__main__':
    main()

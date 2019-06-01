import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, RFE, VarianceThreshold, RFECV

import plot_factory


def remove_features_with_low_variance(x, threshold=0.8):
    selection = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    return selection.fit_transform(x)


def show_correlation_matrix_with_heatmap(x, y):
    x_and_y_train = pd.concat([x, y], axis=1)
    correlation_matrix = x_and_y_train.corr()
    top_corr_features = correlation_matrix.nlargest(20, 'label').index
    sns.heatmap(x_and_y_train[top_corr_features].corr(), cmap="RdYlGn")
    plt.show()


def select_features_by_univariate_selection(x, y, k=20):
    best_feature_selector = SelectKBest(k=k)
    fit = best_feature_selector.fit(x, y)
    fit_scores = pd.DataFrame(fit.scores_)
    fit_columns = pd.DataFrame(x.columns)
    feature_scores = pd.concat([fit_columns, fit_scores], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    return feature_scores


def select_features_by_recursive_feature_elimination(model, x, y, k=20):
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe_fit = rfe.fit(x, y)
    selected_features = x.columns[rfe_fit.support_]
    print(f"Recursive Feature Elimination selected {rfe_fit.n_features_} features")
    print(f"Selected features: {selected_features}")
    print(f"Feature ranking: {rfe_fit.ranking_}")
    return selected_features


def select_features_by_rfe_with_cv(model, x, y):
    rfe_cv = RFECV(estimator=model, cv=5, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
    rfe_cv_fit = rfe_cv.fit(x, y)
    selected_features = x.columns[rfe_cv_fit.support_]
    plot_factory.plot_rfe(rfe_cv_fit.grid_scores_)
    print("Optimal number of features : %d" % rfe_cv_fit.n_features_)
    print(f"Recursive Feature Elimination selected {rfe_cv_fit.n_features_} features")
    print(f"Selected features: {selected_features}")
    print(f"Feature ranking: {rfe_cv_fit.ranking_}")
    return selected_features

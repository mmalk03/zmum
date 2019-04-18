import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.linear_model import LogisticRegression


def univariate_selection(x_train, y_train):
    best_feature_selector = SelectKBest(k=20)
    fit = best_feature_selector.fit(x_train, y_train)
    fit_scores = pd.DataFrame(fit.scores_)
    fit_columns = pd.DataFrame(x_train.columns)
    feature_scores = pd.concat([fit_columns, fit_scores], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    print(feature_scores.nlargest(10, 'Score'))


def feature_importance(x_train, y_train):
    model_extra_trees = ExtraTreesClassifier()
    model_extra_trees.fit(x_train, y_train)
    feature_importances = pd.DataFrame(model_extra_trees.feature_importances_)
    feature_columns = pd.DataFrame(x_train.columns)
    feature_importances_df = pd.concat([feature_columns, feature_importances], axis=1)
    feature_importances_df.columns = ['Feature', 'Importance']

    print(feature_importances_df.nlargest(20, 'Importance'))
    feature_importances_df.nlargest(200, 'Importance').plot(kind='barh')
    plt.show()
    feature_importances_df.nlargest(100, 'Importance').plot(kind='barh')
    plt.show()
    feature_importances_df.nlargest(50, 'Importance').plot(kind='barh')
    plt.show()
    feature_importances_df.nlargest(32, 'Importance').plot(kind='barh')
    plt.show()
    # 32 features with the highest importance will be chosen
    selected_features = list(feature_importances_df.nlargest(32, 'Importance').Feature)


def correlation_matrix_with_heatmap(x_train, y_train):
    x_and_y_train = pd.concat([x_train, y_train], axis=1)
    correlation_matrix = x_and_y_train.corr()
    top_corr_features = correlation_matrix.nlargest(20, 'class').index
    sns.heatmap(x_and_y_train[top_corr_features].corr(), cmap="RdYlGn")
    plt.show()
    # Conclusion: there aren't any features clearly correlated with 'class'


def principal_component_analysis(x_train, y_train):
    pca = PCA(n_components=50)
    pca_fit = pca.fit(x_train)
    print(f"Explained variance: {pca_fit.explained_variance_ratio_}")
    print(pca_fit.components_)
    # Conclusion: ??


def recursive_feature_elimination(x_train, y_train):
    model_logistic_regression = LogisticRegression()
    rfe = RFE(model_logistic_regression, 20)
    rfe_fit = rfe.fit(x_train, y_train)
    print(f"Recursive Feature Elimination selected {rfe_fit.n_features_} features")
    selected_features = x_train.columns[rfe_fit.support_]
    print(f"Selected features: {selected_features}")
    print(f"Feature ranking: {rfe_fit.ranking_}")
    # Conclusion: only features created from categorical ones were chosen - seems a bit suspicious
    # Maybe performing RFE on some other model than Logistic Regression would yield better results

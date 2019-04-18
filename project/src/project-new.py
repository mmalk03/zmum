import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import get_dummies
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor


def save_results(y_test, model_name):
    author_name = 'MIKMAŁ'
    filename = f"../data/{author_name}-{model_name}.txt"
    with open(filename, 'w') as f:
        [f.write(f"{y}\n") for y in [f"\"{author_name}\""] + y_test]


def missing_data(data):
    num_of_null_values = data.isnull().sum()
    percent_of_null_values = num_of_null_values / (data.isnull().count()) * 100
    tt = pd.concat([num_of_null_values, percent_of_null_values], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return tt
    # return np.transpose(tt)


def missing_data_row(data):
    num_of_null_values = data.isnull().sum(axis=1)
    percent_of_null_values = num_of_null_values / (data.isnull().count(axis=1)) * 100
    tt = pd.concat([num_of_null_values, percent_of_null_values], axis=1, keys=['Total', 'Percent'])
    return tt


def build_uniqueness_df(data):
    return pd.DataFrame({'Uniques': data.nunique().sort_values()})


def uniqueness(data, threshold):
    df = build_uniqueness_df(data)
    df[df['Uniques'] < threshold][['Uniques']].plot(kind='hist')
    plt.show()
    features = df[df['Uniques'] < threshold]
    num_features = len(features)
    print(f"{num_features} out of {len(data.columns)} of features have less than {threshold} unique values")
    for f in features.index:
        print(f"{f}: {data[f].unique()}")


def plot_feature_scatter(df1, df2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(4, 4, i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show()


def plot_feature_distribution(df1, df2, label1, label2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5, 5, figsize=(18, 18))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(5, 5, i)
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()


def plot_new_feature_distribution(df1, df2, label1, label2, features):
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2, 4, figsize=(18, 22))

    i = 0
    for feature in features:
        i += 1
        plt.subplot(2, 4, i)
        sns.kdeplot(df1[feature], bw=0.5, label=label1)
        sns.kdeplot(df2[feature], bw=0.5, label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show()


# Load data
train_df = pd.read_csv('project/data/train.txt', sep=' ')
test_df = pd.read_csv('project/data/testx.txt', sep=' ')
print(f"Data shape: train - {train_df.shape}, test - {test_df.shape}")
print('Train head:')
train_df.head()
print('Test head:')
test_df.head()

train_df.describe()
train_df.info()
train_df.corr()
test_df.describe()
test_df.info()

# ***** Analysis of missing data *****
# Lets see missing data on a histogram
missing_data_train_df = missing_data(train_df)
missing_data_train_df['dataset'] = 'train'
missing_data_test_df = missing_data(test_df)
missing_data_test_df['dataset'] = 'test'
missing_data_df = pd.concat([missing_data_train_df, missing_data_test_df])
missing_data_df.Percent = missing_data_df.Percent.astype(int)

sns.countplot(x='Percent', hue='dataset', data=missing_data_df)
plt.show()
# TODO consider showing it on a barplot, where y=variable_name, x=percentage

print(f"There are {sum(missing_data_train_df.Percent == 100)} columns completely filled with NAs in train dataset")
print(f"There are {sum(missing_data_test_df.Percent == 100)} columns completely filled with NAs in test dataset")
empty_columns_train = missing_data_train_df[missing_data_train_df.Percent == 100].index
empty_columns_test = missing_data_test_df[missing_data_test_df.Percent == 100].index
are_columns_the_same = list(empty_columns_train == empty_columns_test).count(False) == 0
print(f"Names of these columns are the same in train and test dataset: {are_columns_the_same}")
# Lets remove these features from both datasets
train_df = train_df.drop(columns=empty_columns_train)
test_df = test_df.drop(columns=empty_columns_test)
print(f"After reduction the train set has {train_df.shape[1]} features")
print(f"After reduction the test set has {test_df.shape[1]} features")

print(f"There are {sum(missing_data_train_df.Percent == 0)} columns without NAs in train dataset")
print(f"There are {sum(missing_data_test_df.Percent == 0)} columns without NAs in test dataset")
full_columns_train = missing_data_train_df[missing_data_train_df.Percent == 0].index
full_columns_test = missing_data_test_df[missing_data_test_df.Percent == 0].index
are_columns_the_same = list(empty_columns_train == empty_columns_test).count(False) == 0
print(f"Names of these columns are the same in train and test dataset: {are_columns_the_same}")

# Lets remove columns with large amount of NAs
remove_features_with_large_amount_of_missing_values = True
if remove_features_with_large_amount_of_missing_values:
    missing_data_train_df = missing_data(train_df)
    removal_threshold = 20  # features missing more than 20% data will be removed
    columns_to_remove_train = list(missing_data_train_df[missing_data_train_df.Percent >= removal_threshold].index)
    columns_to_remove_test = list(missing_data_test_df[missing_data_test_df.Percent >= removal_threshold].index)
    print(f"There are {len(columns_to_remove_train)} columns missing {removal_threshold}% of data in train dataset")
    print(f"There are {len(columns_to_remove_test)} columns missing {removal_threshold}% of data in test dataset")
    # Lets remove these features from both datasets
    train_df = train_df.drop(columns=columns_to_remove_train)
    test_df = test_df.drop(columns=columns_to_remove_train)
    print(f"After reduction the train set has {train_df.shape[1]} features")
    print(f"After reduction the test set has {test_df.shape[1]} features")

# Lets analyse the uniqueness of the data
uniqueness(train_df, 40000)
uniqueness(train_df, 5000)
uniqueness(train_df, 1000)
uniqueness(train_df, 100)
uniqueness(train_df, 20)
uniqueness(train_df, 10)

# Are there any columns with only 1 unique value?
train_uniqueness_df = build_uniqueness_df(train_df)
print(f"There are {sum(train_uniqueness_df.Uniques == 1)} columns with only 1 unique value in train dataset")
single_unique_columns_train = train_uniqueness_df[train_uniqueness_df.Uniques == 1].index
# Lets remove these features from both datasets
train_df = train_df.drop(columns=single_unique_columns_train)
test_df = test_df.drop(columns=single_unique_columns_train)
print(f"After reduction the train set has {train_df.shape[1]} features")
print(f"After reduction the test set has {test_df.shape[1]} features")

# Lets choose categorical features
print(f"Train set has following feature types: {set(train_df.dtypes)}")
print(f"Test set has following feature types: {set(test_df.dtypes)}")
df = pd.DataFrame(train_df.nunique().sort_values())
categorical_features = list(train_df.select_dtypes(['O']).columns)
treat_numerical_features_with_small_uniqueness_as_categorical = False
if treat_numerical_features_with_small_uniqueness_as_categorical:
    categoricality_threshold = 20  # all columns with less than 20 unique values will be treated as categorical
    categorical_features = list(set(
        categorical_features +
        list(df[df[0] < categoricality_threshold].index)
    ))
print(f"Number of features classified as categorical is {len(categorical_features)}")

# And now, remove those which have a lot of missing values - those are probably just text columns
uniquality_threshold = 100
features_to_remove = [feature for feature in categorical_features
                      if train_uniqueness_df.transpose()[feature][0] > uniquality_threshold]
print(f"There are {len(features_to_remove)} categorical features having more than {uniquality_threshold} unique values")
train_df = train_df.drop(columns=features_to_remove)
test_df = test_df.drop(columns=features_to_remove)
[categorical_features.remove(feature) for feature in features_to_remove]
print(f"After reduction the train set has {train_df.shape[1]} features")
print(f"After reduction the test set has {test_df.shape[1]} features")

# ***** Data imputation *****
# Lets replace all categorical missing values with the most frequent occurring element
for feature in categorical_features:
    use_most_common_value_for_replacement = False
    if use_most_common_value_for_replacement:
        value_for_nas = train_df[feature].value_counts().index[0]  # TODO should this also include values from test set?
    else:
        value_for_nas = 'unknown'
    train_df.loc[train_df[feature].isna(), feature] = value_for_nas
    test_df.loc[test_df[feature].isna(), feature] = value_for_nas

train_df[categorical_features] = train_df[categorical_features].astype('category')
test_df[categorical_features] = test_df[categorical_features].astype('category')

# Lets replace all numerical missing values with median of given column
numerical_features = set(train_df.columns[:-1]) - set(categorical_features)
print(f"Number of features classified as numerical is {len(numerical_features)}")
for feature in numerical_features:
    median = train_df[feature].median()  # TODO should this also include values from test set?
    train_df.loc[train_df[feature].isna(), feature] = median
    test_df.loc[test_df[feature].isna(), feature] = median

print(f"Train and test datasets have in total {train_df.isnull().sum().sum() + test_df.isnull().sum().sum()} NAs")

# ***** One-hot encoding *****

train_len = len(train_df)
y_train = train_df.pop('class')
merged_df = pd.concat([train_df, test_df], axis=0)
merged_df_with_dummies = pd.get_dummies(merged_df)
x_train = merged_df_with_dummies[:train_len]
x_test = merged_df_with_dummies[train_len:]

# After one-hot encoding some columns from the train dataset may have only 1 unique value
# this will happen if for some feature the test dataset has values not present in train dataset
train_uniqueness_df = build_uniqueness_df(x_train)
print(f"There are {sum(train_uniqueness_df.Uniques == 1)} columns with only 1 unique value in train dataset")
single_unique_columns_train = train_uniqueness_df[train_uniqueness_df.Uniques == 1].index
# Lets remove these features from both datasets
x_train = x_train.drop(columns=single_unique_columns_train)
x_test = x_test.drop(columns=single_unique_columns_train)
print(f"After one-hot encoding train and test datasets have {len(x_train.columns)} columns")

# ********** END OF DATA PRE-PROCESSING **********

# Lets present some of the features on scatter plots
features = list(x_train.transpose().index[0:16])
plot_feature_scatter(x_train[0:len(x_test)], x_test, features)

plot_feature_scatter(x_train[:1000], x_test[:1000], features)
features = list(x_train.transpose().index[16:32])
plot_feature_scatter(x_train[:1000], x_test[:1000], features)

# Density plots of features

# Firstly lets analyse distribution for values with target value 0 and 1
t0 = x_train.loc[y_train == 0]
t1 = x_train.loc[y_train == 1]
features = x_train.select_dtypes(['float64', 'int64']).columns[:-1]
features = features[0:25]
plot_feature_distribution(t0, t1, '0', '1', features)

# We can observe that some of the features are clearly different depending on 'class'
# Those features are: Var38, Var57, Var73, Var76, Var133, Var134, Var153
# Especially Var153

# Lets now compare features from train and test data sets
features = x_train.select_dtypes(['float64', 'int64']).columns[:-1]
plot_feature_distribution(x_train, x_test, 'train', 'test', features[0:25])
# Train and test set seems to be well balanced with respect to the distribution of numeric variables

# Features correlation

# Lets analyse the correlations between the features in train set
features = x_train.select_dtypes(['float64', 'int64']).columns[:-1]
# features = x_train.columns[:-1]
correlations = x_train[features].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]

# Top most correlated features, besides the same feature pairs
correlations.tail(20)
# There are features which are highly correlated

# Top least correlated features, besides the same feature pairs
correlations.head(10)

# Duplicated values
features = x_train.columns[:-1]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = x_train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = x_test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])

# Lets show the top 15 max of duplicate values per train set
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value']))
             .sort_values(by='Max duplicates', ascending=False)
             .head(15)).transpose()
# And for the test set
np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value']))
             .sort_values(by='Max duplicates', ascending=False)
             .head(15)).transpose()

# ********** Feature selection **********

# Univariate selection

best_feature_selector = SelectKBest(k=20)
fit = best_feature_selector.fit(x_train, y_train)
fit_scores = pd.DataFrame(fit.scores_)
fit_columns = pd.DataFrame(x_train.columns)
feature_scores = pd.concat([fit_columns, fit_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores.nlargest(10, 'Score'))

# Feature importance
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

# Correlation matrix with heatmap
x_and_y_train = pd.concat([x_train, y_train], axis=1)
correlation_matrix = x_and_y_train.corr()
top_corr_features = correlation_matrix.nlargest(20, 'class').index
sns.heatmap(x_and_y_train[top_corr_features].corr(), cmap="RdYlGn")
plt.show()
# Conclusion: there aren't any features clearly correlated with 'class'

# Principal Component Analysis
pca = PCA(n_components=50)
pca_fit = pca.fit(x_train)
print(f"Explained variance: {pca_fit.explained_variance_ratio_}")
print(pca_fit.components_)
# Conclusion: ??

# Recursive Feature Elimination using Logistic Regression (takes a lot of time)
use_rfe = False
if use_rfe:
    model_logistic_regression = LogisticRegression()
    recursive_feature_elimination = RFE(model_logistic_regression, 20)
    rfe_fit = recursive_feature_elimination.fit(x_train, y_train)
    print(f"Recursive Feature Elimination selected {rfe_fit.n_features_} features")
    selected_features = x_train.columns[rfe_fit.support_]
    print(f"Selected features: {selected_features}")
    print(f"Feature ranking: {rfe_fit.ranking_}")
    # Conclusion: only features created from categorical ones were chosen - seems a bit suspicious
    # Maybe performing RFE on some other model than Logistic Regression would yield better results

# ********** MODEL **********

# XGBoost

xgboost_model = XGBRegressor()
xgboost_model.fit(x_train, y_train)

y_test_predictions = xgboost_model.predict(x_test)
save_results(list(y_test_predictions), 'XGBoost')

y_train_pred = xgboost_model.predict(x_train)
auc = roc_auc_score(y_train, y_train_pred)
print(f"auc on train set: {auc}")

y_test_pred_sorted = list(reversed(sorted(y_test_predictions)))
chosen_results = y_test_pred_sorted[:round(0.1 * len(y_test_pred_sorted))]
print(f"10% of predicted probabilities are greater than {chosen_results[-1]}")

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
    max_features=None,  # number of features to consider when looking for the best split; None: max_features=n_features
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
    max_features=None,  # number of features to consider when looking for the best split; None: max_features=n_features
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

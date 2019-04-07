import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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


def uniqueness(data, threshold):
    df = pd.DataFrame(data.nunique().sort_values())
    df[df[0] < threshold][0].plot(kind='hist')
    plt.show()
    features = df[df[0] < threshold]
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

# Lets analyse the uniqueness of the data
temp = train_df.describe().transpose()
uniqueness(train_df, 40000)
uniqueness(train_df, 5000)
uniqueness(train_df, 1000)
uniqueness(train_df, 100)
uniqueness(train_df, 10)
uniqueness(train_df, 20)

# ***** Data imputation *****
print(f"Train set has following feature types: {set(train_df.dtypes)}")
print(f"Test set has following feature types: {set(test_df.dtypes)}")
# Lets replace all categorical missing values with the most frequent occurring element
categoricality_threshold = 20  # all columns with less than 20 unique values will be treated as categorical
df = pd.DataFrame(train_df.nunique().sort_values())
categorical_features = set(list(df[df[0] < categoricality_threshold].index) + list(train_df.select_dtypes(['O']).columns))
print(f"Number of features classified as categorical is {len(categorical_features)}")
for feature in categorical_features:
    most_common_value = train_df[feature].value_counts().index[0]
    train_df.loc[train_df[feature].isna(), feature] = most_common_value
    # TODO should NAs from the test set also be removed?

category_features = train_df.select_dtypes('O').columns
train_df[category_features ] = train_df[category_features ].astype('category')
test_df[category_features ] = test_df[category_features ].astype('category')

# Lets replace all numerical missing values with median of given column
numerical_features = set(train_df.columns[:-1]) - categorical_features
print(f"Number of features classified as numerical is {len(numerical_features)}")
for feature in numerical_features:
    train_df.loc[train_df[feature].isna(), feature] = train_df[feature].median()
    # TODO should NAs from the test set also be removed?

# Lets present some of the features on scatter plots
features = list(train_df.transpose().index[0:16])
plot_feature_scatter(train_df[0:len(test_df)], test_df, features)

plot_feature_scatter(train_df[:1000], test_df[:1000], features)
features = list(train_df.transpose().index[16:32])
plot_feature_scatter(train_df[:1000], test_df[:1000], features)

# Density plots of features

# Firstly lets analyse distribution for values with target value 0 and 1
t0 = train_df.loc[train_df['class'] == 0]
t1 = train_df.loc[train_df['class'] == 1]
features = train_df.select_dtypes(['float64', 'int64']).columns[:-1]
plot_feature_distribution(t0, t1, '0', '1', features)

# We can observe that some of the features are clearly different depending on 'class'
# Those features are: Var38, Var57, Var73, Var76, Var133, Var134, Var153
# Especially Var153

# Lets now compare features from train and test data sets
features = train_df.select_dtypes(['float64', 'int64']).columns[:-1]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)

# Train and test set seems to be well balanced with respect to the distribution of numeric variables


# Features correlation

# Lets analyse the correlations between the features in train set
features = train_df.select_dtypes(['float64', 'int64']).columns[:-1]
# features = train_df.columns[:-1]
correlations = train_df[features].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]

# Top most correlated features, besides the same feature pairs
correlations.tail(10)
# There are features which are highly correlated

# Top least correlated features, besides the same feature pairs
correlations.head(10)

# Duplicated values
features = train_df.columns[:-1]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train_df[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test_df[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])

# Lets show the top 15 max of duplicate values per train set
np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value']))
             .sort_values(by='Max duplicates', ascending=False)
             .head(15)).transpose()
# And for the test set
np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value']))
             .sort_values(by='Max duplicates', ascending=False)
             .head(15))

# Unique values
uniques_train = []
for feature in features:
    uniques_train.append([feature, train_df[feature].nunique()])

uniques_train_df = (pd.DataFrame(uniques_train, columns=['Feature', 'Uniques count'])
                    .sort_values(by='Uniques count', ascending=False))
categorical_features = uniques_train_df[uniques_train_df['Uniques count'] < 100].Feature
train_df[categorical_features] = train_df[categorical_features].astype('category')
test_df[categorical_features] = test_df[categorical_features].astype('category')

# temp = train_df.select_dtypes('object').describe().transpose()
# categorical_features = temp[temp.unique < 100].index
# train_df[categorical_features] = train_df[categorical_features].astype('category')
# test_df[categorical_features] = test_df[categorical_features].astype('category')
# Model
features = train_df.select_dtypes(['float64', 'int64', 'category']).columns.values[:-1]
target = train_df['class']
params = {
    'num_leaves': 6,
    'max_bin': 63,
    'min_data_in_leaf': 45,
    'learning_rate': 0.01,
    'min_sum_hessian_in_leaf': 0.000446,
    'bagging_fraction': 0.55,
    'bagging_freq': 5,
    'max_depth': 14,
    'save_binary': True,
    'seed': 31452,
    'feature_fraction_seed': 31415,
    'feature_fraction': 0.51,
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
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (train_idx, valid_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    train_data = lgb.Dataset(train_df.iloc[train_idx][features], label=target.iloc[train_idx])
    valid_data = lgb.Dataset(train_df.iloc[valid_idx][features], label=target.iloc[valid_idx])

    num_round = 15000
    clf = lgb.train(params, train_data, num_round,
                    valid_sets=[train_data, valid_data],
                    verbose_eval=1000,
                    early_stopping_rounds=250)
    oof[valid_idx] = clf.predict(train_df.iloc[valid_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df['Feature'] = features
    fold_importance_df['Importance'] = clf.feature_importance()
    fold_importance_df['Fold'] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
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

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

# Look for missing data
missing_data_df = missing_data(train_df)
missing_data_df.Percent.plot(kind='hist')
plt.show()
# A very basic histogram shows that there is quite a lot of features which have more than 90% of missing data
num_of_features_missing_more_than_90 = missing_data_df[missing_data_df['Percent'] > 90]
print(f"There are exactly {len(num_of_features_missing_more_than_90)} variables which miss more than 90% of data")

# Similar situation can be observed in the test set
test_missing_data_df = missing_data(test_df)
test_missing_data_df.Percent.plot(kind='hist')
plt.show()
# TODO show both histograms on the same plot for comparison

# What about other columns with big number of na's?
f1 = missing_data_df.Percent < 90
f2 = missing_data_df.Percent > 30
n2 = missing_data_df[f1 & f2]
print(f"There are {len(n2)} variables which miss between 30% and 90% of data")

# TODO before removal it would be nice to see how do they correlate with target class,
#  maybe some of them are really important, regardless of the number of NAs
# Lets remove them completely
train_cols_to_drop = missing_data_df[missing_data_df['Percent'] > 30].index
train_df = train_df.drop(columns=train_cols_to_drop)
test_df = test_df.drop(columns=train_cols_to_drop)
print(f"After reduction the train set has {train_df.shape[1]} features")
missing_data_df = missing_data(train_df)
missing_data_df.Percent.plot(kind='hist')
plt.show()

# Lets now analyse NAs row-wise
missing_data_row_df = missing_data_row(train_df)
missing_data_row_df.Percent.plot(kind='hist')
plt.show()
# It looks like some of the rows are missing more than 50% of the data, lets remove these rows from the training set
# TODO it may be better to firstly fill in data in columns and than do this kind of things
rows_to_drop = missing_data_row_df[missing_data_row_df.Percent >= 50].index
train_df = train_df.drop(index=rows_to_drop)

# Lets now reduce the number of features even further - we will take only those for which every row has no NAs
md1 = missing_data(train_df)
md1[md1.Percent <= 5].Percent.plot(kind='hist')
plt.show()
print(f"There are {len(md1[md1.Percent == 0])} features for which each row doesn't have any NAs")
train_df = train_df.drop(columns=md1[md1.Percent != 0].index)
print(f"We are left with {train_df.shape[0]} training values and {train_df.shape[1]} features")

# Lets do the same for the test set
test_missing_data_row_df = missing_data_row(test_df)
test_missing_data_row_df.Percent.plot(kind='hist')
plt.show()
rows_to_drop = test_missing_data_row_df[test_missing_data_row_df.Percent >= 50].index
test_df = test_df.drop(index=rows_to_drop)
test_df = test_df.drop(columns=md1[md1.Percent != 0].index)
print(f"We are left with {test_df.shape[0]} test values")

len(train_df.transpose().index)
len(test_df.transpose().index)

for train, test in zip(train_df.transpose().index, test_df.transpose().index):
    if train != test:
        print(f"{train} : {test}")

train_df.describe()
train_df.info()
train_df.corr()
test_df.describe()
test_df.info()

# Lets present some of the features on scatter plots
features = list(train_df.transpose().index[0:16])
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

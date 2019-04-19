import numpy as np
import pandas as pd

import data_processing as data_processing
import plot_factory as plot_factory

x_train, y_train, x_test = data_processing.prepare_data()

# Lets present some of the features on scatter plots
features = list(x_train.transpose().index[0:16])
plot_factory.plot_feature_scatter(x_train[0:len(x_test)], x_test, features)

plot_factory.plot_feature_scatter(x_train[:1000], x_test[:1000], features)
features = list(x_train.transpose().index[16:32])
plot_factory.plot_feature_scatter(x_train[:1000], x_test[:1000], features)

# Density plots of features

# Firstly lets analyse distribution for values with target value 0 and 1
t0 = x_train.loc[y_train == 0]
t1 = x_train.loc[y_train == 1]
features = x_train.select_dtypes(['float64', 'int64']).columns[:-1]
features = features[0:25]
plot_factory.plot_feature_distribution(t0, t1, '0', '1', features)

# We can observe that some of the features are clearly different depending on 'class'
# Those features are: Var38, Var57, Var73, Var76, Var133, Var134, Var153
# Especially Var153

# Lets now compare features from train and test data sets
features = x_train.select_dtypes(['float64', 'int64']).columns[:-1]
plot_factory.plot_feature_distribution(x_train, x_test, 'train', 'test', features[0:25])
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

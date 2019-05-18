import matplotlib.pyplot as plt
import pandas as pd

import data_repository as repository
import plot_factory as plot_factory


def basic_data_frame_analysis(df):
    print(f"Shape: {df.shape}")
    print(f"Head: {df.head()}")
    df.describe()
    df.info()
    df.corr()


def analyse_uniqueness(data):
    df = pd.DataFrame({'Uniques': data.nunique().sort_values()})
    df[['Uniques']].plot(kind='hist')
    plt.xlabel('Number of unique values')
    plt.ylabel('Number of columns')
    plt.show()


def get_baseline_results():
    return [1] * 600, [i for i in range(10)]


x_train, y_train = repository.load_train_dataset()
x_test = repository.load_test_dataset()

# Basic analysis
basic_data_frame_analysis(x_train)
basic_data_frame_analysis(x_test)

# Unique values
analyse_uniqueness(x_train)
analyse_uniqueness(x_test)

# Scatter plots of features
for i in range(len(x_train.columns) // 16):
    plot_factory.plot_feature_scatter(x_train[0:len(x_test)], x_test, x_train.columns[i:i + 16])

# Density plots of features

# Depending on label
first = x_train.loc[(y_train == 1)['label']]
second = x_train.loc[(y_train == -1)['label']]
for i in range(len(x_train.columns) // 20):
    plot_factory.plot_feature_distribution(first, second, '0', '1', x_train.columns[i:i + 20])

# Depending on data set
for i in range(len(x_train.columns) // 20):
    plot_factory.plot_feature_distribution(x_train, x_test, 'train', 'test', x_train.columns[i:i + 20])

# Features correlation in train set
correlations = x_train.corr().abs().unstack().sort_values(kind='quicksort').reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.tail(20)
correlations.head(20)

# Example prediction
y_test, features_test = get_baseline_results()
repository.save_results('All 1s', y_test, features_test)

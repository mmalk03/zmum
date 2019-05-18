import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import data_repository as repository


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


def basic_data_frame_analysis(df):
    print(f"Shape: {df.shape}")
    print(f"Head: {df.head()}")
    df.describe()
    df.info()
    df.corr()


def show_missing_data_histograms(train_df, test_df):
    # Lets see missing data on a histogram
    missing_data_train_df = missing_data(train_df)
    missing_data_train_df['dataset'] = 'train'
    missing_data_test_df = missing_data(test_df)
    missing_data_test_df['dataset'] = 'test'
    missing_data_df = pd.concat([missing_data_train_df, missing_data_test_df])
    missing_data_df.Percent = missing_data_df.Percent.astype(int)

    fig = plt.figure()
    sns.countplot(x='Percent', hue='dataset', data=missing_data_df)
    plt.show()


def analyse_columns_without_nas(train_df, test_df):
    missing_data_train_df = missing_data(train_df)
    missing_data_test_df = missing_data(test_df)
    print(f"There are {sum(missing_data_train_df.Percent == 0)} columns without NAs in train dataset")
    print(f"There are {sum(missing_data_test_df.Percent == 0)} columns without NAs in test dataset")
    full_columns_train = missing_data_train_df[missing_data_train_df.Percent == 0].index
    full_columns_test = missing_data_test_df[missing_data_test_df.Percent == 0].index
    are_columns_the_same = list(full_columns_train == full_columns_test).count(False) == 0
    print(f"Names of these columns are the same in train and test dataset: {are_columns_the_same}")


def remove_columns_completely_filled_with_nas(train_df, test_df):
    missing_data_train_df = missing_data(train_df)
    missing_data_test_df = missing_data(test_df)
    print(f"There are {sum(missing_data_train_df.Percent == 100)} columns completely filled with NAs in train dataset")
    print(f"There are {sum(missing_data_test_df.Percent == 100)} columns completely filled with NAs in test dataset")
    empty_columns_train = missing_data_train_df[missing_data_train_df.Percent == 100].index
    empty_columns_test = missing_data_test_df[missing_data_test_df.Percent == 100].index
    are_columns_the_same = list(empty_columns_train == empty_columns_test).count(False) == 0
    print(f"Names of these columns are the same in train and test dataset: {are_columns_the_same}")
    # Lets remove these features from both datasets
    train_df.drop(columns=empty_columns_train, inplace=True)
    test_df.drop(columns=empty_columns_test, inplace=True)
    print(f"After reduction the train set has {train_df.shape[1]} features")
    print(f"After reduction the test set has {test_df.shape[1]} features")


def remove_columns_with_nas(train_df, test_df, removal_threshold=90):
    missing_data_train_df = missing_data(train_df)
    missing_data_test_df = missing_data(test_df)
    columns_to_remove_train = list(missing_data_train_df[missing_data_train_df.Percent >= removal_threshold].index)
    columns_to_remove_test = list(missing_data_test_df[missing_data_test_df.Percent >= removal_threshold].index)
    print(f"There are {len(columns_to_remove_train)} columns missing {removal_threshold}% of data in train dataset")
    print(f"There are {len(columns_to_remove_test)} columns missing {removal_threshold}% of data in test dataset")
    train_df.drop(columns=columns_to_remove_train, inplace=True)
    test_df.drop(columns=columns_to_remove_train, inplace=True)
    print(f"After reduction the train set has {train_df.shape[1]} features")
    print(f"After reduction the test set has {test_df.shape[1]} features")


def analyse_uniqueness(train_df):
    uniqueness(train_df, 40000)
    uniqueness(train_df, 5000)
    uniqueness(train_df, 1000)
    uniqueness(train_df, 100)
    uniqueness(train_df, 20)
    uniqueness(train_df, 10)


def remove_columns_with_1_unique_value(train_df, test_df):
    train_uniqueness_df = build_uniqueness_df(train_df)
    print(f"There are {sum(train_uniqueness_df.Uniques == 1)} columns with only 1 unique value in train dataset")
    single_unique_columns_train = train_uniqueness_df[train_uniqueness_df.Uniques == 1].index
    train_df.drop(columns=single_unique_columns_train, inplace=True)
    test_df.drop(columns=single_unique_columns_train, inplace=True)
    print(f"After reduction the train set has {train_df.shape[1]} features")
    print(f"After reduction the test set has {test_df.shape[1]} features")


def mark_features_as_categorical(train_df, test_df, treat_numerical_with_small_uniqueness_as_categorical=False):
    print(f"Train set has following feature types: {set(train_df.dtypes)}")
    print(f"Test set has following feature types: {set(test_df.dtypes)}")
    df = pd.DataFrame(train_df.nunique().sort_values())
    categorical_features = list(train_df.select_dtypes(['O']).columns)
    if treat_numerical_with_small_uniqueness_as_categorical:
        categoricality_threshold = 20  # all columns with less than 20 unique values will be treated as categorical
        categorical_features = list(set(
            categorical_features +
            list(df[df[0] < categoricality_threshold].index)
        ))
    print(f"Number of features classified as categorical is {len(categorical_features)}")
    train_df[categorical_features] = train_df[categorical_features].astype('category')
    test_df[categorical_features] = test_df[categorical_features].astype('category')


def remove_text_columns(train_df, test_df, uniquality_threshold=100):
    categorical_features = list(train_df.select_dtypes(['category']).columns)
    train_uniqueness_df = build_uniqueness_df(train_df)
    features_to_remove = [feature for feature in categorical_features
                          if train_uniqueness_df.transpose()[feature][0] > uniquality_threshold]
    print(f"{len(features_to_remove)} categorical features have more than {uniquality_threshold} unique values")
    train_df.drop(columns=features_to_remove, inplace=True)
    test_df.drop(columns=features_to_remove, inplace=True)
    print(f"After reduction the train set has {train_df.shape[1]} features")
    print(f"After reduction the test set has {test_df.shape[1]} features")


def impute_categorical_features(train_df, test_df, use_most_common_value_for_replacement=False):
    categorical_features = list(train_df.select_dtypes(['category']).columns)
    for feature in categorical_features:
        if use_most_common_value_for_replacement:
            # TODO should this also include values from test set?
            value_for_nas = train_df[feature].value_counts().index[0]
        else:
            value_for_nas = 'unknown'
            train_df[feature].cat.add_categories(value_for_nas, inplace=True)
            test_df[feature].cat.add_categories(value_for_nas, inplace=True)
        train_df[feature].fillna(value_for_nas, inplace=True)
        test_df[feature].fillna(value_for_nas, inplace=True)
        # train_df.loc[train_df[feature].isna(), feature] = value_for_nas
        # test_df.loc[test_df[feature].isna(), feature] = value_for_nas


def impute_numerical_features(train_df, test_df):
    categorical_features = list(train_df.select_dtypes(['category']).columns)
    numerical_features = set(train_df.columns[:-1]) - set(categorical_features)
    print(f"Number of features classified as numerical is {len(numerical_features)}")
    for feature in numerical_features:
        median = train_df[feature].median()  # TODO should this also include values from test set?
        train_df.loc[train_df[feature].isna(), feature] = median
        test_df.loc[test_df[feature].isna(), feature] = median


def one_hot_encoding(train_df, test_df):
    train_len = len(train_df)

    train_df_temp = train_df
    y_train = train_df_temp.pop('class')

    merged_df = pd.concat([train_df_temp, test_df], axis=0)
    merged_df_with_dummies = pd.get_dummies(merged_df)
    train_df = pd.concat([merged_df_with_dummies[:train_len], y_train], axis=1)
    test_df = merged_df_with_dummies[train_len:]

    # After one-hot encoding some columns from the train dataset may have only 1 unique value
    # this will happen if for some feature the test dataset has values not present in train dataset
    remove_columns_with_1_unique_value(train_df, test_df)
    print(f"After one-hot encoding train dataset has {len(train_df.columns)} columns")
    print(f"After one-hot encoding test dataset has {len(test_df.columns)} columns")
    return train_df, test_df


def prepare_data():
    train_df = repository.load_train_dataset()
    test_df = repository.load_test_dataset()
    basic_data_frame_analysis(train_df)
    basic_data_frame_analysis(test_df)

    show_missing_data_histograms(train_df, test_df)
    analyse_columns_without_nas(train_df, test_df)
    remove_columns_completely_filled_with_nas(train_df, test_df)
    remove_columns_with_nas(train_df, test_df)

    analyse_uniqueness(train_df)

    remove_columns_with_1_unique_value(train_df, test_df)

    mark_features_as_categorical(train_df, test_df)

    remove_text_columns(train_df, test_df)
    impute_categorical_features(train_df, test_df)
    impute_numerical_features(train_df, test_df)
    train_df, test_df = one_hot_encoding(train_df, test_df)

    y_train = train_df.pop('class')
    return train_df, y_train, test_df

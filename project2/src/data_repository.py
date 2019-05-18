import os

import pandas as pd


def load_train_dataset():
    data = pd.read_csv('project2/data/artificial_train.data', sep=' ', header=None)
    df = pd.DataFrame(data=data, columns=list(range(500)))
    df.columns = [c + 1 for c in df.columns]
    labels = pd.read_csv('project2/data/artificial_train.labels', header=None)
    labels.columns = ['label']
    return df, labels


def load_test_dataset():
    data = pd.read_csv('project2/data/artificial_valid.data', sep=' ', header=None)
    df = pd.DataFrame(data=data, columns=list(range(500)))
    df.columns = [c + 1 for c in df.columns]
    return df


def save_results(model_name, predictions, features):
    save_dir = f"project2/results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/MIKMAŁ_artificial_prediction.txt", 'w') as f:
        f.write('"MIKMAŁ"\n')
        [f.write(f"{p}\n") for p in predictions]
    with open(f"{save_dir}/MIKMAŁ_artificial_features.txt", 'w') as f:
        f.write('"MIKMAŁ"\n')
        [f.write(f"{feature}\n") for feature in features]

import pandas as pd


def load_train_dataset():
    return pd.read_csv('project/data/train.txt', sep=' ')


def load_test_dataset():
    return pd.read_csv('project/data/testx.txt', sep=' ')


def save_results(y_test, model_name):
    author_name = 'MIKMAŁ'
    filename = f"../data/{author_name}-{model_name}.txt"
    with open(filename, 'w') as f:
        [f.write(f"{y}\n") for y in [f"\"{author_name}\""] + y_test]
